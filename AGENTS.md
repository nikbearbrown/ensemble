# AGENTS.md: ensemble

For Codex and other coding agents. This folder is a set of skills and a command-line tool for machine-learning ensembles (combined predictive models, not groups of agents).

When pointed at a dataset and asked which models to try, or asked to build a model, ensemble, blend, stack or average models, combine forecasts, or review ensemble code: read `SKILL.md` in this folder and follow it. It points to `skills/choose-models/SKILL.md`, `skills/build-ensemble/SKILL.md` and `skills/review-ensemble/SKILL.md`. The rules are `docs/rules.md`; the file layout is `docs/file-format.md`. `CLAUDE.md` lists the six things that matter most; they apply to every agent.

Commands: `./ens help`. Python 3.9 or newer and numpy; pandas and scikit-learn for `profile`, `suggest` and `build`. After changing `scripts/ens.py`: `python3 -m unittest discover tests`.
