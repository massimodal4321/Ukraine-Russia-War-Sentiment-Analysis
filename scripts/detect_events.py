"""
Event detector for the narrative compass (docs/compass/events.json).

Runs once a day after the page is rebuilt. It looks for news stories that are
suddenly carried by far more outlets than normal, asks Gemini whether each one is
a single significant event in the war, names it in the house style, and adds it
to the events file as a live label. Each new label also opens a GitHub issue with
the headlines behind it, so it can be checked.

What decides inclusion is how widely an event is reported, never how far the
index moved.

It only ever adds entries. It never edits or deletes one, so any label changed by
hand stays changed, and any label set to "hidden" stays hidden and also stops the
same event from being added again.

    python scripts/detect_events.py              normal daily run
    python scripts/detect_events.py --dry-run    show what it would add, change nothing
    python scripts/detect_events.py --backtest   replay the free first stage over all history
"""
import collections
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT     = pathlib.Path(__file__).resolve().parents[1]
DAYS_DIR = ROOT / "data" / "days"
SITE     = ROOT / "docs" / "index.html"
EVENTS   = ROOT / "docs" / "compass" / "events.json"
STATE    = ROOT / "data" / ".detector_state.json"
MODEL    = os.environ.get("MODEL", "gemini-3.5-flash-lite")

# ---- stage 1: free, deterministic, runs every day -------------------------------
WINDOW      = 3     # days a story has to build up its coverage
MIN_OUTLETS = 15    # distinct outlets that must carry the same name inside the window
RATIO       = 4.0   # and at least this many times that name's normal coverage
MIN_EXCESS  = 60    # outlets above normal that a story needs before it is sent on
BASE_FROM   = 60    # "normal" is the average from 60 days ...
BASE_TO     = 7     # ... to 7 days before the window
SHARE       = 0.25  # names sharing this share of headlines belong to one story
LOOKBACK    = 7     # re-check the last week, so a skipped run loses nothing
REJUDGE     = 14    # a story already sent to the model is not sent again for this long
SAME_DAYS   = 3     # an existing label this close is taken to be the same event
NEAR_DAYS   = 21    # labels this close are shown to the model to rule out duplicates

# ---- stage 2: the model's verdict, and the house rules -------------------------
MAX_PER_7   = 1     # at most one new label in any 7 days
MAX_PER_30  = 2     # and two in any 30
MAX_LABEL   = 36
EVIDENCE_N  = 40    # headlines shown to the model per story

WORD = re.compile(r"[^\W\d_][\w'\u2019\-]{3,}", re.UNICODE)


def names_in(title):
    """Capitalised words of four letters or more, in any script, lowercased."""
    out = set()
    for w in WORD.findall(title):
        w = re.sub(r"['\u2019]s$", "", w)
        if len(w) >= 4 and w[0].isupper():
            out.add(w.lower())
    return out


def ordinal(d):
    return dt.date.fromisoformat(d).toordinal()


def iso(o):
    return dt.date.fromordinal(o).isoformat()


# ---------------------------------------------------------------- loading
class Index:
    """Every stored headline, plus which outlets carried which name on which day."""

    def __init__(self, start="2021-12-01"):
        self.heads = []                    # (ordinal, title, outlet, lang, url)
        self.day_names = {}                # ordinal -> {name: set(outlets)}
        self.day_heads = collections.defaultdict(list)
        for f in sorted(DAYS_DIR.glob("*.json")):
            d = f.stem[:10]
            if d < start:
                continue
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            o = ordinal(d)
            per = collections.defaultdict(set)
            for x in (j.get("scored") or []) + (j.get("dropped") or []):
                t, dom = x.get("t"), x.get("d")
                if not t or not dom:
                    continue
                i = len(self.heads)
                self.heads.append((o, t, dom, x.get("l", ""), x.get("h", "")))
                self.day_heads[o].append(i)
                for n in names_in(t):
                    per[n].add(dom)
            self.day_names[o] = per
        self.last = max(self.day_names) if self.day_names else None

    def daily_outlets(self, name, o):
        return len(self.day_names.get(o, {}).get(name, ()))

    def normal(self, name, start):
        days = range(start - BASE_FROM, start - BASE_TO)
        return sum(self.daily_outlets(name, o) for o in days) / len(days)


def stories(ix, end):
    """Stories whose coverage in the 3 days ending on `end` is far above normal."""
    span = [end - k for k in range(WINDOW)]
    if any(o not in ix.day_names for o in span):
        return []                         # only judge complete windows
    rough = collections.Counter()
    for o in span:
        for n, doms in ix.day_names[o].items():
            rough[n] += len(doms)
    hot = {}
    for n, c in rough.items():
        if c < MIN_OUTLETS:
            continue
        outlets = set().union(*(ix.day_names[o].get(n, set()) for o in span))
        if len(outlets) < MIN_OUTLETS:
            continue
        base = WINDOW * ix.normal(n, span[-1])
        if len(outlets) >= RATIO * (base + 1):
            hot[n] = (len(outlets), base)
    if not hot:
        return []
    heads = {n: {i for o in span for i in ix.day_heads[o] if n in names_in(ix.heads[i][1])}
             for n in hot}
    # names that keep appearing in the same headlines are one story
    parent = {n: n for n in hot}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    ns = sorted(hot)
    for a in range(len(ns)):
        for b in range(a + 1, len(ns)):
            x, y = heads[ns[a]], heads[ns[b]]
            if len(x & y) >= SHARE * min(len(x), len(y)):
                parent[root(ns[a])] = root(ns[b])
    groups = collections.defaultdict(list)
    for n in hot:
        groups[root(n)].append(n)
    out = []
    for members in groups.values():
        members.sort(key=lambda n: -hot[n][0])
        lead = members[0]
        hs = sorted(set().union(*(heads[n] for n in members)))
        outlets = {ix.heads[i][2] for i in hs}
        excess = len(outlets) - hot[lead][1]
        if excess < MIN_EXCESS:
            continue
        first = min(o for o in span if ix.daily_outlets(lead, o) > 0)
        out.append({"names": members, "lead": lead, "end": end, "start": first,
                    "outlets": len(outlets), "excess": round(excess, 1),
                    "langs": len({ix.heads[i][3] for i in hs}), "heads": hs})
    return sorted(out, key=lambda s: -s["excess"])


# ---------------------------------------------------------------- existing labels
def load_events():
    if EVENTS.exists():
        return json.loads(EVENTS.read_text(encoding="utf-8"))
    return {"events": []}


def all_labels(doc):
    """Every label the compass knows about, including hidden ones and the main site's."""
    out = [(ordinal(e["date"]), e["label"]) for e in doc.get("events", [])
           if e.get("date") and e.get("label")]
    try:
        t = SITE.read_text(encoding="utf-8")
        s = t.index("const D = ")
        e = t.index(";\n", s)
        for x in json.loads(t[s + 10:e]).get("events", []):
            out.append((ordinal(x["d"]), x["label"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    return out


# ---------------------------------------------------------------- the model
PROMPT = """You are labelling events on a chart of how news media covered the Russia-Ukraine war.

The headlines below, from {start} to {end} and in many languages, share the names: {names}.
Decide whether they report ONE concrete event in the Russia-Ukraine war that happened on a specific day.

Significance:
- major: a city or region changes hands; a major offensive starts or ends; a strike that changes a side's
  capabilities; the top military or political leadership changes; another country's forces join the fighting.
- notable: a very large strike; a major aid, weapons or sanctions decision; a summit or formal talks between
  the leaders; an incident involving a NATO country; a mutiny or coup attempt.
- minor: everything else, including routine fighting, casualty claims, warnings, threats, predictions, opinion,
  analysis, speculation, anniversaries, holidays, truce offers and stories not about this war.

Label rules: at most {max_label} characters. Say who did what, in plain English for someone who knows nothing
about the war. Use the proper name of an operation, place or person where there is one. No dates, no quotation
marks, neutral wording. If one side claims something the other disputes, say "claims".
Good labels: Russia captures Avdiivka | Ukraine's Kursk incursion | Ukraine's Operation Spiderweb |
Russia claims capture of Pokrovsk | Wagner mutiny in Russia | North Korea sends troops to Russia

Labels already on the chart near these dates:
{nearby}

Reply with JSON only, no other text:
{{"single_event": true or false, "war_related": true or false, "significance": "major" or "notable" or "minor",
"event_date": "YYYY-MM-DD", "label": "...", "key_name": "the place, operation or person the label is about",
"same_as": null or the exact text of one label from the list above if it is the same event}}

Headlines:
{headlines}"""


def pick_evidence(ix, story, n=EVIDENCE_N):
    """A spread of headlines: one per outlet first, all languages, earliest first."""
    seen, out = set(), []
    for i in sorted(story["heads"], key=lambda i: ix.heads[i][0]):
        if ix.heads[i][2] in seen:
            continue
        seen.add(ix.heads[i][2])
        out.append(i)
        if len(out) >= n:
            break
    return out


def ask_model(ix, story, nearby):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    ev = pick_evidence(ix, story)
    listing = "\n".join(f"{k+1}. [{iso(ix.heads[i][0])}] ({ix.heads[i][3]}, {ix.heads[i][2]}) {ix.heads[i][1]}"
                        for k, i in enumerate(ev))
    prompt = PROMPT.format(start=iso(story["start"]), end=iso(story["end"]),
                           names=", ".join(story["names"][:6]), max_label=MAX_LABEL,
                           nearby="\n".join(f"- {iso(o)}: {l}" for o, l in nearby) or "- none",
                           headlines=listing)
    r = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
    txt = (r.text or "").replace("```json", "").replace("```", "")
    m = re.search(r"\{[\s\S]*\}", txt)
    return json.loads(m.group(0)) if m else {}


def check(ix, story, v, labels):
    """House rules the model's answer must pass. Returns (ok, reason)."""
    if not v.get("single_event"):
        return False, "not a single event"
    if not v.get("war_related"):
        return False, "not about the war"
    if v.get("significance") not in ("major", "notable"):
        return False, f"significance: {v.get('significance')}"
    if v.get("same_as"):
        return False, f"same as existing label: {v['same_as']}"
    label = (v.get("label") or "").strip()
    if not label or len(label) > MAX_LABEL or re.search(r"[\"\u201c\u201d]", label) or not label[0].isupper():
        return False, f"label breaks the rules: {label!r}"
    try:
        o = ordinal(v.get("event_date", ""))
    except ValueError:
        return False, "no valid date"
    if not story["start"] - 3 <= o <= story["end"]:
        return False, f"date {iso(o)} outside the story"
    if any(abs(o - lo) <= SAME_DAYS for lo, _ in labels):
        return False, "an existing label is within 3 days"
    key = (v.get("key_name") or "").lower().strip()
    hs = [i for i in story["heads"] if key and key in ix.heads[i][1].lower()]
    if len({ix.heads[i][2] for i in hs}) < 3 and not (set(key.split()) & set(story["names"])):
        return False, f"key name {key!r} not found in the headlines"
    return True, ""


# ---------------------------------------------------------------- the run
def notify(entry, story, ix):
    body = [f"The detector added **{entry['label']}** on {entry['date']} "
            f"({'major' if entry['tier'] == 1 else 'notable'}) to `docs/compass/events.json`.",
            "", f"Reported by {story['outlets']} outlets in {story['langs']} languages "
            f"within {WINDOW} days, against about {round(story['outlets'] - story['excess'])} normally.",
            "", "Headlines behind it:"]
    body += [f"- [{e['title']}]({e['url']}) ({e['outlet']})" for e in entry["evidence"]]
    body += ["", "If it is wrong, edit its `label`, or set its `status` to `hidden`. "
             "Do not delete the entry, or it may be added again."]
    try:
        subprocess.run(["gh", "issue", "create", "--title",
                        f"New compass label: {entry['label']} ({entry['date']})",
                        "--body", "\n".join(body)], check=True, capture_output=True, timeout=60)
        print("  issue opened")
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  could not open an issue ({e}); the label is still live")


def run(dry=False):
    stems = sorted(f.stem[:10] for f in DAYS_DIR.glob("*.json"))
    if not stems:
        print("no data")
        return
    # only the last few months are needed: the window, the lookback and the "normal" baseline
    ix = Index(start=iso(ordinal(stems[-1]) - BASE_FROM - WINDOW - LOOKBACK - 5))
    if ix.last is None:
        print("no data")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {"judged": []}
    doc = load_events()
    labels = all_labels(doc)
    added = 0
    for end in range(ix.last - LOOKBACK + 1, ix.last + 1):
        for s in stories(ix, end):
            if any(set(s["names"]) & set(j["names"]) and abs(end - ordinal(j["end"])) <= REJUDGE
                   for j in state["judged"]):
                continue
            if any(s["start"] - SAME_DAYS <= lo <= s["end"] + SAME_DAYS for lo, _ in labels):
                continue
            autos = [ordinal(e["date"]) for e in doc["events"] if e.get("origin") == "auto"]
            if (sum(1 for a in autos if abs(a - end) < 7) >= MAX_PER_7 or
                    sum(1 for a in autos if abs(a - end) < 30) >= MAX_PER_30):
                print(f"{iso(end)} {s['lead']}: limit of new labels reached, skipped")
                continue
            print(f"{iso(end)} story '{s['lead']}' ({s['outlets']} outlets, +{s['excess']} above normal)")
            nearby = [(o, l) for o, l in labels if abs(o - s["start"]) <= NEAR_DAYS]
            if dry and not os.environ.get("GEMINI_KEY"):
                print("  dry run without a key: would ask the model")
                continue
            try:
                v = ask_model(ix, s, nearby)
            except Exception as e:                       # the model failing must never block the data
                print(f"  model call failed: {e}")
                continue
            ok, why = check(ix, s, v, labels)
            state["judged"].append({"names": s["names"][:8], "end": iso(end),
                                    "verdict": "added" if ok else why, "label": v.get("label")})
            if not ok:
                print(f"  rejected: {why}")
                continue
            entry = {"date": v["event_date"], "label": v["label"].strip(),
                     "tier": 1 if v["significance"] == "major" else 2,
                     "origin": "auto", "status": "live", "added": dt.date.today().isoformat(),
                     "evidence": [{"title": ix.heads[i][1], "url": ix.heads[i][4], "outlet": ix.heads[i][2]}
                                  for i in pick_evidence(ix, s, 5)]}
            print(f"  ADDED: {entry['label']} ({entry['date']})")
            if dry:
                continue
            doc["events"].append(entry)
            doc["events"].sort(key=lambda e: e.get("date", ""))
            labels.append((ordinal(entry["date"]), entry["label"]))
            added += 1
            EVENTS.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            notify(entry, s, ix)
    state["judged"] = state["judged"][-400:]
    if not dry:
        STATE.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n")
    print(f"done, {added} label(s) added")


def backtest():
    """Replays stage 1 over all history, with a stand-in for the model, to count stories."""
    ix = Index()
    labels = all_labels(load_events())
    judged, found = [], []
    for end in range(ordinal("2022-03-15"), ix.last + 1):
        for s in stories(ix, end):
            if any(set(s["names"]) & set(j["names"]) and end - j["end"] <= REJUDGE for j in judged):
                continue
            judged.append({"names": s["names"], "end": end})
            found.append(s)
    yrs = (ix.last - ordinal("2022-03-15")) / 365.25
    hit = [l for o, l in labels if any(s["start"] - SAME_DAYS <= o <= s["end"] + SAME_DAYS for s in found)]
    print(f"{len(found)} stories would go to the model ({len(found)/yrs:.1f} a year); "
          f"{len(set(hit))} of {len(set(labels))} existing labels have one")
    for s in found:
        near = [l for o, l in labels if s["start"] - SAME_DAYS <= o <= s["end"] + SAME_DAYS]
        print(f"{iso(s['start'])} {s['outlets']:4d} outlets +{s['excess']:6.1f} {s['langs']:3d} langs  "
              f"{', '.join(s['names'][:4]):48s} {'= ' + near[0] if near else ''}")


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest()
    else:
        run(dry="--dry-run" in sys.argv)
