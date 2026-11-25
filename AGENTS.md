# AGENTS.md: ensemble

For Codex and other coding agents. This folder is a set of skills and a command-line tool for machine-learning ensembles (combined predictive models, not groups of agents).

When asked to ensemble, blend, stack or average models, combine forecasts, or review ensemble code: read `SKILL.md` in this folder and follow it. It points to `skills/build-ensemble/SKILL.md` and `skills/review-ensemble/SKILL.md`. The rules are `docs/rules.md`; the file layout is `docs/file-format.md`. `CLAUDE.md` lists the five things that matter most; they apply to every agent.

Commands: `./ens help`. Python 3.9 or newer and numpy. After changing `scripts/ens.py`: `python3 -m unittest discover tests`.
