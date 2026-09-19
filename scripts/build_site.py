"""
build_site.py — refresh the site's embedded data from data/daily.json.

The page is one self-contained HTML file with the dataset baked into it, which is why
it loads instantly, works offline, and can be emailed to someone. The cost is that it
has to be rebuilt when the data changes. This does that, and nothing else.

It is not a template renderer. It reads the existing docs/index.html, swaps out the one
payload line, and writes it back. Everything else in the page — layout, styling, the
curated event list, every fix — is left exactly as it is. So there is no second copy of
the site to keep in step, and editing the page by hand stays safe.

    python scripts/build_site.py

Run it after the backfill commits. It exits 0 and changes nothing if the data has not
moved, so it is safe to run on every job.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "daily.json"
SITE = ROOT / "docs" / "index.html"

INVASION = "2022-02-24"          # the series starts here; earlier days are pre-war
KINDS = ["territory", "position", "capability", "strike", "attrition"]
SOURCES = ["publication", "belligerent", "analyst"]

# The classifier occasionally returns a label outside its own list. These two are
# plainly the right category with the wrong spelling, so they are folded in; anything
# else is dropped rather than guessed at.
FOLD = {"ATTRITION": "attrition", " strike": "strike"}


def build_points(daily: dict) -> list[dict]:
    """Everything the page draws, in the shape it expects."""
    out = []
    for day in sorted(k for k in daily if k >= INVASION):
        r = daily[day]

        kinds = {c: 0 for c in KINDS}
        for name, n in r["kinds"].items():
            c = FOLD.get(name, name)
            if c in kinds:
                kinds[c] += n

        by_kind = {}
        for c in KINDS:
            src = r["by_kind"].get(c)
            if src is None:                      # try the misspelt form before zeroing
                for bad, good in FOLD.items():
                    if good == c and bad in r["by_kind"]:
                        src = r["by_kind"][bad]
                        break
            by_kind[c] = ({"ua": round(src["ua"], 1), "ru": round(src["ru"], 1), "n": src["n"]}
                          if src else {"ua": 0, "ru": 0, "n": 0})

        out.append({
            "d": day,
            "ua": round(r["ua"], 1), "ru": round(r["ru"], 1),
            "uaB": round(r["ua_nobell"], 1), "ruB": round(r["ru_nobell"], 1),
            "n": r["n"], "sampled": r["sampled"],
            "nlangs": len(r["langs"]),
            "by_kind": by_kind, "kinds": kinds,
            "srcs": {c: r["srcs"].get(c, 0) for c in SOURCES},
        })
    return out


def find_gaps(points: list[dict]) -> list[list[str]]:
    """Runs of dates with no score at all. The page breaks its lines across these and
    refuses to average over them, so they have to be exact."""
    have = {p["d"] for p in points}
    first = dt.date.fromisoformat(points[0]["d"])
    last = dt.date.fromisoformat(points[-1]["d"])
    gaps, run = [], []
    for i in range((last - first).days + 1):
        day = (first + dt.timedelta(days=i)).isoformat()
        if day not in have:
            run.append(day)
        elif run:
            gaps.append([run[0], run[-1]])
            run = []
    if run:
        gaps.append([run[0], run[-1]])
    return gaps


def main() -> int:
    if not DATA.exists():
        sys.exit(f"no dataset at {DATA}")
    if not SITE.exists():
        sys.exit(f"no page at {SITE}. Commit the built page there once, by hand, and "
                 f"this keeps it current from then on.")

    src = json.loads(DATA.read_text(encoding="utf-8"))
    html = SITE.read_text(encoding="utf-8")

    start = html.index("const D = ")
    end = html.index(";\n", start)
    current = json.loads(html[start + 10:end])

    points = build_points(src["daily"])
    if not points:
        sys.exit("no scored days at or after the invasion; refusing to write an empty page")

    payload = {
        "pts": points,
        # The event marks are written by hand and are not in the dataset, so they are
        # carried across untouched. Losing them to a rebuild would be easy to miss.
        "events": current["events"],
        "gaps": find_gaps(points),
        "meta": {k: src["meta"][k] for k in
                 ("rubric", "model", "per_day", "temp", "domain_cap")} |
                {"updated": src["meta"]["updated"][:10]},
    }

    was = len(current["pts"])
    now = len(points)
    if json.dumps(payload["pts"]) == json.dumps(current["pts"]):
        print(f"data unchanged at {was} days, nothing to write")
        return 0

    html = html[:start + 10] + json.dumps(payload, separators=(",", ":")) + html[end:]

    # the provenance line quotes the coverage, so it has to move with the data
    html = re.sub(
        r"(score, from )\d{4}-\d{2}-\d{2}( to )\d{4}-\d{2}-\d{2}",
        rf"\g<1>{points[0]['d']}\g<2>{points[-1]['d']}", html, count=1)
    html = re.sub(r"[\d,]+ days carry a\s*\n?\s*score",
                  f"{now:,} days carry a\n  score", html, count=1)
    html = re.sub(r"Backfill last written \d{4}-\d{2}-\d{2}\.",
                  f"Backfill last written {payload['meta']['updated']}.", html, count=1)

    SITE.write_text(html, encoding="utf-8")
    gaps = ", ".join(f"{a} to {b}" for a, b in payload["gaps"]) or "none"
    print(f"rebuilt {SITE.relative_to(ROOT)}: {was} -> {now} days, "
          f"{points[0]['d']} to {points[-1]['d']}")
    print(f"  gaps: {gaps}")
    print(f"  events carried over: {len(payload['events'])}")
    print(f"  page is {len(html) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
