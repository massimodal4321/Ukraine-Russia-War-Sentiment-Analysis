# Narrative Index: Ukraine and Russia

How strongly world news coverage claimed each side was gaining, since January 2022.

Measures what the press asserted, not what happened on the ground.

## Method

For each day, up to 400 headlines are drawn from GDELT's Global Knowledge Graph,
restricted to coverage naming both countries under an armed-conflict theme,
deduplicated, and capped at three articles per news domain so a single wire story
cannot dominate a day.

Each headline is scored by `gemma-4-31b-it` at temperature 0 against a fixed
rubric on four fields: whether it makes a specific claim about either side's
current position, how strongly it portrays Ukraine gaining, how strongly it
portrays Russia gaining, and whether the claim is attributed to a named party.

The two scores are independent and do not sum to 100. Both low means stalemate.
Both high means a contested narrative.

## Known limitations

- Headlines only, not article bodies
- Weights by story prominence as well as claim strength, since a wire story
  running in twelve languages contributes twelve times
- GDELT's source mix has shifted over time. Russian-domain outlets fell from
  about 18% of this coverage before February 2022 to about 4% by early 2023
- Scores are coarse. The classifier resolves to roughly four levels per side
- Validated against two events: Avdiivka, 17 Feb 2024 (ua 2.5, ru 72.9) and the
  Kharkiv counteroffensive, 11 Sep 2022 (ua 56.1, ru 22.9)

## Running

Actions tab, Backfill workflow, Run workflow. Pick a phase:

| Phase | Days | Approx hours |
|---|---|---|
| monthly | 56 | 1.5 |
| 10daily | 171 | 4.5 |
| weekly | 245 | 6.4 |
| every3rd | 570 | 15 |
| daily | 1710 | 45 |

Each run works for about 5 hours 20 minutes, commits, then triggers itself again
if days remain. Safe to stop at any point.

Changing `scripts/rubric.txt` invalidates every stored score. The script checks
the rubric hash and refuses to run on a mismatch.
