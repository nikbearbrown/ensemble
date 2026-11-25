# ensemble

Skills and a tool for machine-learning ensembles. Read `SKILL.md` first.

The five things that matter most:

1. `min_gain` is set by the user before any ensemble score exists, and is not changed to fit a result.
2. Anything learned about the combination (members, weights, stacker, calibrator, threshold) is learned from out-of-fold predictions on development rows.
3. A score on rows that chose the recipe is a diagnostic. Say so every time you quote one.
4. `./ens verdict` runs once per project, on one locked recipe. Never re-run it to see whether a changed recipe passes; never edit `holdout-ledger.json` or delete entries from `RECORD.md`.
5. Report what was run and what was not. With no data or no baseline, deliver an evaluation plan, not results.

Working on this repo: the tool is `scripts/ens.py` (numpy only). Run `python3 -m unittest discover tests` after any change. If a rule number changes in `docs/rules.md`, change it in the skills and in the tool's messages too.
