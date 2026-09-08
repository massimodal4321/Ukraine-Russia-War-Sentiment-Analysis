#!/usr/bin/env python3
"""
Narrative Index backfill / daily update.

Runs in GitHub Actions. Resumable: reads data/daily.json, works out which days
are missing, and processes as many as it can inside a time budget before
committing and exiting. Chain runs to cover the whole range.

Env:
  GEMINI_KEY        Gemini API key
  GCP_SA_KEY_FILE   path to service account JSON
  PHASE             monthly | 10daily | weekly | every3rd | daily
  BUDGET_MIN        minutes to work before stopping cleanly (default 320)
"""
import collections, datetime as dt, html, json, os, re, statistics, sys, time, unicodedata
from pathlib import Path

from google.cloud import bigquery
from google import genai
from google.genai import types
import py3langid as langid

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data" / "daily.json"
import hashlib
RUBRIC    = (ROOT / "scripts" / "rubric.txt").read_text()
# normalise line endings and trailing spaces so a browser paste cannot change the hash
RUBRIC    = "\n".join(l.rstrip() for l in
            RUBRIC.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()
RUBRIC_HASH = hashlib.sha256(RUBRIC.encode()).hexdigest()[:12]

MODEL       = os.environ.get("MODEL", "gemini-3.5-flash-lite")
TEMPERATURE = 0
# 400 tested at 400/400 correct with no positional drift. Long generations do
# occasionally truncate, corrupt an object, or draw a 500 from the API, so the parser
# below salvages whatever came back rather than discarding the request. Drop to 200 if
# the log ever shows failures on a large share of days.
BATCH       = int(os.environ.get("BATCH", "400"))
PER_DAY     = 400
# 400 objects measured at 11,893 output tokens with compact keys; this rubric's longer
# keys run higher, so the cap is set well clear of it. 8192 was the old value and it
# silently truncated the response mid-array.
MAX_OUT     = 24000
DOMAIN_CAP  = 3
START       = "2022-01-01"
PROJECT     = "ukraine-russia-sentiment"
PHASE       = os.environ.get("PHASE", "monthly")
BUDGET      = int(os.environ.get("BUDGET_MIN", "320")) * 60
GAP         = int(os.environ.get("GAP_S", "12"))   # seconds between requests.
# At batch 400 a request carries roughly 19k input and up to 18k output tokens. Five a
# minute is about 74% of the 250k tokens-a-minute ceiling, which leaves room for a retry
# without tripping the limit. Eight seconds would sit at 111% and fail.

STEP = {"monthly": None, "10daily": 10, "weekly": 7, "every3rd": 3, "daily": 1}[PHASE]

REQ_TIMEOUT_MS = int(os.environ.get("REQ_TIMEOUT_S", "150")) * 1000
client = genai.Client(api_key=os.environ["GEMINI_KEY"],
                      http_options=types.HttpOptions(timeout=REQ_TIMEOUT_MS))
bq     = bigquery.Client.from_service_account_json(
             os.environ["GCP_SA_KEY_FILE"], project=PROJECT)

# ---------------------------------------------------------------- cleaning
SEP         = re.compile(r'(\s*[|\u2022\u00b7\u00bb]\s*|\s+[\u2013\u2014-]\s+)[^|\u2022\u00b7\u2013\u2014]{3,60}$')
SITE_HEAD   = re.compile(r'^[A-Za-z0-9.\- ]{3,30}\s+-\s+')
SEO_PREFIX  = re.compile(r'^[a-z0-9 ]{3,40}:\s+(?=[A-Z])')
BARE_DOMAIN = re.compile(r'\s+[A-Za-z0-9\-]{2,20}\.(ua|ru|com|net|org|eu|pl|de|info|co\.uk)\s*$', re.I)

def clean_title(t):
    if not t:
        return None
    t = ''.join(c for c in t.strip() if unicodedata.category(c) != 'Cf')
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1]
    t = html.unescape(html.unescape(t.replace('""', '"')))
    t = t.replace('\u2014', ' ').replace('\ufffd', '')
    t = BARE_DOMAIN.sub('', SITE_HEAD.sub('', t))
    for _ in range(2):
        n = SEP.sub('', t)
        if n == t or len(n) < 25:
            break
        t = n
    return re.sub(r'\s+', ' ', SEO_PREFIX.sub('', t)).strip(' -|\u2022')

def key(t):
    return re.sub(r'[^\w ]', '', t.lower(), flags=re.UNICODE).strip()

# ---------------------------------------------------------------- retrieval
SQL = """
WITH raw AS (
  SELECT DocumentIdentifier url, SourceCommonName source,
         REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') title
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE(_PARTITIONTIME) = @day
    AND V2Locations LIKE '%Ukraine%' AND V2Locations LIKE '%Russia%'
    AND V2Themes LIKE '%ARMEDCONFLICT%'
),
ded AS (
  SELECT ANY_VALUE(url) url, ANY_VALUE(source) source, ANY_VALUE(title) title
  FROM raw WHERE title IS NOT NULL
  GROUP BY LOWER(REGEXP_REPLACE(title, r'[^a-zA-Z0-9 ]', ''))
),
capped AS (
  SELECT *, ROW_NUMBER() OVER
    (PARTITION BY source ORDER BY FARM_FINGERPRINT(CONCAT(url, 'cap'))) rn FROM ded
)
-- Deterministic sampling. RAND() drew a different 400 headlines on every run, so the
-- same date scored differently each time and the dataset could not be reproduced.
-- FARM_FINGERPRINT gives a stable pseudo-random order keyed to the article URL, so the
-- same 400 are chosen every time. The two salts keep the per-domain cap and the final
-- pick from correlating with each other.
SELECT url, source, title FROM capped WHERE rn <= @cap
ORDER BY FARM_FINGERPRINT(CONCAT(url, 'pick')) LIMIT @lim
"""

def pull(day):
    print(f"  {day}: querying BigQuery...", flush=True)
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("day", "DATE", day),
        bigquery.ScalarQueryParameter("cap", "INT64", DOMAIN_CAP),
        bigquery.ScalarQueryParameter("lim", "INT64", PER_DAY * 3)])
    out, seen = [], set()
    for r in bq.query(SQL, job_config=cfg).result():
        c = clean_title(r["title"])
        if not c or len(c) < 20:
            continue
        k = key(c)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append({"title": c, "source": r["source"], "url": r["url"],
                    "lang": langid.classify(c)[0]})
    print(f"  {day}: {len(out)} clean headlines", flush=True)
    return out[:PER_DAY]

# ---------------------------------------------------------------- scoring
class DeadKey(Exception):
    pass


class QuotaOut(Exception):
    pass

def score(batch, retries=4):
    listing = "\n".join(f"{i+1}. {h['title']}" for i, h in enumerate(batch))
    for a in range(retries):
        t_req = time.time()
        try:
            r = client.models.generate_content(
                model=MODEL, contents=RUBRIC + "\n\nHeadlines:\n" + listing,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE, max_output_tokens=MAX_OUT))
            txt = (r.text or "").replace("```json", "").replace("```", "")
            # One malformed object used to throw away the whole day and burn a retry.
            # Try the array first; if it will not parse, take every object that will,
            # and report what broke so the cause is visible in the log.
            arr, salvaged = None, False
            m = re.search(r"\[[\s\S]*\]", txt)
            if m:
                try:
                    arr = json.loads(m.group(0))
                except json.JSONDecodeError as je:
                    at = getattr(je, "pos", 0)
                    print(f"      malformed JSON at char {at}: "
                          f"{m.group(0)[max(0,at-70):at+70]!r}", flush=True)
            if arr is None:
                arr = []
                for o in re.findall(r"\{[^{}]*\}", txt):
                    try:
                        arr.append(json.loads(o))
                    except Exception:
                        pass
                salvaged = True
                if not arr:
                    raise ValueError("no parseable objects in the response")
            out = {}
            for o in arr:
                i = int(o.get("i", 0)) - 1
                if 0 <= i < len(batch):
                    out[batch[i]["title"]] = {
                        "url": batch[i]["url"], "dom": batch[i]["source"],
                        "rel": bool(o.get("relevant")), "ua": float(o.get("ua", 0)),
                        "ru": float(o.get("ru", 0)), "attr": bool(o.get("attributed")),
                        "kind": str(o.get("kind", "other")),
                        "src": str(o.get("source", "publication")),
                        "lang": batch[i]["lang"]}
            note = f", salvaged {len(arr)} of {len(batch)}" if salvaged else ""
            print(f"      ok in {time.time()-t_req:.0f}s{note}", flush=True)
            return out
        except Exception as e:
            msg = str(e)
            print(f"      failed after {time.time()-t_req:.0f}s", flush=True)
            # never retry an auth failure, waiting cannot fix it
            if "401" in msg or "403" in msg or "UNAUTHENTICATED" in msg:
                raise DeadKey(msg[:200])
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                raise QuotaOut(msg[:200])
            print(f"      retry {a+1}/{retries}: {msg[:110]}", flush=True)
            time.sleep(8 * (a + 1))
    return {}

def score_day(items):
    """Returns (scores, complete). Incomplete days are never written."""
    S, complete = {}, True
    nb = (len(items) + BATCH - 1) // BATCH
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        print(f"    batch {i//BATCH + 1}/{nb} ...", flush=True)
        got = score(chunk)
        if len(got) < len(chunk):
            time.sleep(15)
            retry = score(chunk)
            got = retry if len(retry) > len(got) else got
        if len(got) < len(chunk) * 0.9:
            complete = False
            print(f"      SHORT {len(got)}/{len(chunk)}", flush=True)
        S.update(got)
        print(f"    batch {i//BATCH + 1}/{nb} -> {len(got)}/{len(chunk)}", flush=True)
        time.sleep(GAP)
    return S, complete

# ---------------------------------------------------------------- days
def day_list():
    d0, d1 = dt.date.fromisoformat(START), dt.date.today() - dt.timedelta(days=1)
    if STEP is None:
        out, y, m = [], d0.year, d0.month
        while dt.date(y, m, 15) <= d1:
            if dt.date(y, m, 15) >= d0:
                out.append(dt.date(y, m, 15).isoformat())
            m += 1
            if m == 13:
                m, y = 1, y + 1
        return out
    out, n = [], d0
    while n <= d1:
        out.append(n.isoformat())
        n += dt.timedelta(days=STEP)
    return out

# ---------------------------------------------------------------- main
def main():
    q = ROOT / "data" / ".quota_stopped"
    if q.exists():
        q.unlink()
    # A record of what this run achieved. The chain refuses to fire when it is zero,
    # which makes a runaway loop of fast-failing runs impossible.
    (ROOT / "data" / ".progress").write_text("0")
    print("checking connections...", flush=True)
    try:
        n = list(bq.query("SELECT 1 AS ok").result())[0].ok
        print(f"  BigQuery OK ({n})", flush=True)
    except Exception as e:
        sys.exit(f"BigQuery failed: {str(e)[:200]}")
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=MODEL, contents="Reply with the single word OK",
                config=types.GenerateContentConfig(temperature=0))
            print(f"  Gemini OK ({r.text.strip()[:20]})", flush=True)
            break
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                sys.exit(f"Gemini auth failed: {str(e)[:200]}")
            print(f"  Gemini attempt {attempt+1}/4 failed: {str(e)[:110]}", flush=True)
            if attempt == 3:
                sys.exit("Gemini unreachable after 4 attempts, their end. Try again later.")
            time.sleep(20 * (attempt + 1))

    store = json.loads(DATA.read_text()) if DATA.exists() else {"meta": {}, "daily": {}}
    if store["meta"].get("rubric") and store["meta"]["rubric"] != RUBRIC_HASH:
        sys.exit(f"RUBRIC CHANGED: stored {store['meta']['rubric']} vs current "
                 f"{RUBRIC_HASH}. Old scores are not comparable. Clear data/daily.json "
                 f"to re-score, or restore the old rubric.")

    days = day_list()
    empty = set(store.get("meta", {}).get("empty", []))
    todo = [d for d in days if d not in store["daily"] and d not in empty]
    print(f"phase={PHASE} rubric={RUBRIC_HASH} model={MODEL}")
    print(f"{len(days)} days in range, {len(store['daily'])} done, {len(todo)} to go")
    print(f"budget {BUDGET//60} min\n", flush=True)

    t0, done = time.time(), 0
    for day in todo:
        if time.time() - t0 > BUDGET:
            print(f"\nbudget reached after {done} days, stopping cleanly")
            break
        try:
            items = pull(day)
            if len(items) < 30:
                # GDELT genuinely has no coverage on some dates. Recording it means the
                # date leaves the queue. Leaving it unrecorded kept "remaining" above
                # zero for ever, and the chain kept launching runs to retry a date that
                # can never succeed, each lap costing a partition scan and a request.
                store["meta"].setdefault("empty", [])
                if day not in store["meta"]["empty"]:
                    store["meta"]["empty"].append(day)
                    store["meta"]["empty"].sort()
                    DATA.write_text(json.dumps(store, separators=(",", ":"),
                                               sort_keys=True))
                print(f"{day}  only {len(items)} headlines, recorded as empty",
                      flush=True)
                continue
            S, complete = score_day(items)
            if not complete:
                print(f"{day}  INCOMPLETE, not written", flush=True)
                continue
            rel = [t for t in S if S[t]["rel"]]
            if not rel:
                print(f"{day}  no relevant headlines, not written", flush=True)
                continue
            ua_u = [S[t]["ua"] for t in rel if not S[t]["attr"]]
            ru_u = [S[t]["ru"] for t in rel if not S[t]["attr"]]
            # index excluding belligerent claims but keeping independent analysts
            ua_nb = [S[t]["ua"] for t in rel if S[t]["src"] != "belligerent"]
            ru_nb = [S[t]["ru"] for t in rel if S[t]["src"] != "belligerent"]
            store["daily"][day] = {
                "ua": round(statistics.mean(S[t]["ua"] for t in rel), 2),
                "ru": round(statistics.mean(S[t]["ru"] for t in rel), 2),
                "ua_unattr": round(statistics.mean(ua_u), 2) if ua_u else None,
                "ru_unattr": round(statistics.mean(ru_u), 2) if ru_u else None,
                "ua_nobell": round(statistics.mean(ua_nb), 2) if ua_nb else None,
                "ru_nobell": round(statistics.mean(ru_nb), 2) if ru_nb else None,
                "n": len(rel), "sampled": len(S),
                "attr": sum(1 for t in rel if S[t]["attr"]),
                "langs": dict(collections.Counter(S[t]["lang"] for t in rel).most_common(8)),
                "kinds": dict(collections.Counter(S[t]["kind"] for t in rel).most_common()),
                "srcs": dict(collections.Counter(S[t]["src"] for t in rel).most_common()),
                "by_kind": {k: {"ua": round(statistics.mean(
                                    [S[t]["ua"] for t in rel if S[t]["kind"] == k]), 1),
                                "ru": round(statistics.mean(
                                    [S[t]["ru"] for t in rel if S[t]["kind"] == k]), 1),
                                "n": sum(1 for t in rel if S[t]["kind"] == k)}
                            for k in set(S[t]["kind"] for t in rel)},
            }
            store["meta"] = {"rubric": RUBRIC_HASH, "model": MODEL, "temp": TEMPERATURE,
                             "per_day": PER_DAY, "batch": BATCH, "domain_cap": DOMAIN_CAP,
                             "updated": dt.datetime.now(dt.timezone.utc).isoformat()}
            DATA.write_text(json.dumps(store, separators=(",", ":"), sort_keys=True))

            # The day file: what the site shows when a reader opens this date. Kept in
            # its own file so the main series stays small and only the day someone
            # actually opens is fetched. Dropped headlines are capped, because keeping
            # all 400 a day would run to roughly 100 MB across the whole backfill.
            trim = lambda x, n=180: x[:n]
            row = store["daily"][day]
            scored = [{"t": trim(t), "u": S[t]["ua"], "v": S[t]["ru"], "k": S[t]["kind"],
                       "q": S[t]["src"], "l": S[t]["lang"], "d": S[t]["dom"], "h": S[t]["url"]}
                      for t in sorted(rel, key=lambda t: -(S[t]["ua"] - S[t]["ru"]))]
            dropped = [{"t": trim(t), "d": S[t]["dom"], "l": S[t]["lang"], "h": S[t]["url"]}
                       for t in [x for x in S if not S[x]["rel"]][:25]]
            daydir = ROOT / "data" / "days"
            daydir.mkdir(parents=True, exist_ok=True)
            (daydir / f"{day}.json").write_text(json.dumps(
                {"d": day, "ua": row["ua"], "ru": row["ru"], "n": row["n"],
                 "sampled": row["sampled"], "dropped_total": row["sampled"] - row["n"],
                 "scored": scored, "dropped": dropped}, separators=(",", ":")))

            done += 1
            d = store["daily"][day]
            el = (time.time() - t0) / 60
            print(f"{day}  ua {d['ua']:5.1f}  ru {d['ru']:5.1f}  n={d['n']:3}  "
                  f"[{done}/{len(todo)}] {el:.0f}m elapsed", flush=True)
        except DeadKey as e:
            sys.exit(f"AUTH FAILED, not retrying: {e}")
        except QuotaOut as e:
            print(f"\nDAILY QUOTA REACHED after {done} days. Stopping cleanly.")
            print("Progress is saved. The scheduled run tomorrow will continue.")
            (ROOT / "data" / ".quota_stopped").write_text("1")
            break
        except Exception as e:
            print(f"{day}  ERROR {str(e)[:120]}", flush=True)
            time.sleep(15)

    remaining = len([d for d in days if d not in store["daily"]])
    (ROOT / "data" / ".remaining").write_text(str(remaining))
    (ROOT / "data" / ".progress").write_text(str(done))
    print(f"\n{len(store['daily'])}/{len(days)} days complete for phase {PHASE}")
    print(f"remaining: {remaining}")

if __name__ == "__main__":
    main()
