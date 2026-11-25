# ensemble

Skills and a tool for deciding which machine-learning models to build, building them, and judging whether combining them is worth it. Read `SKILL.md` first.

The six things that matter most:

0. `./ens suggest` is a starting list from the data's shape. Edit it with what you know about the problem, and give every member a one-line reason for what it brings that the others do not.

1. `min_gain` is set by the user before any ensemble score exists, and is not changed to fit a result.
2. Anything learned about the combination (members, weights, stacker, calibrator, threshold) is learned from out-of-fold predictions on development rows.
3. A score on rows that chose the recipe is a diagnostic. Say so every time you quote one.
4. `./ens verdict` runs once per project, on one locked recipe. Never re-run it to see whether a changed recipe passes; never edit `holdout-ledger.json` or delete entries from `RECORD.md`.
5. Report what was run and what was not. With no data or no baseline, deliver an evaluation plan, not results.

Working on this repo: the tool is `scripts/ens.py` (numpy only) and `scripts/ens_plan.py` (pandas); the build script is `templates/build_members.py`. Run `python3 -m unittest discover tests` after any change. If a rule number changes in `docs/rules.md`, change it in the skills and in the tool's messages too.
