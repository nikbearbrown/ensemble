# ensemble

Skills that tell Claude, Codex or any coding agent how to build a machine-learning ensemble in an order that keeps the evaluation honest, and a small command-line tool that says whether the ensemble earned its place.

These are ensembles of predictive models (bagging, boosting, blending, stacking, deep ensembles), not groups of AI agents.

## In short

- An ensemble helps only when competent models make different mistakes. Most that disappoint were not badly combined; they were badly evaluated.
- You save each model's predictions as CSV files keyed by row id. `./ens` checks them, shows whether the models make the same mistakes, compares the best single model with a plain average and with learned weights, locks one recipe, and scores it once on a holdout nothing has touched.
- The answer is one of three words: **KEEP**, **TOO SMALL** or **NOT SHOWN**. Anything but KEEP means ship the single model.
- You decide the smallest gain worth having (`min_gain`) before any ensemble score exists. The tool will not run without it.

## Get started

```bash
git clone https://github.com/nikbearbrown/ensemble
cd ensemble
./ens help
```

See it work on synthetic data (needs scikit-learn):

```bash
./ens demo demo/binary
./ens check demo/binary      # are the files aligned, on one scale, leak-free by fold?
./ens members demo/binary    # each model alone, and whether they make the same mistakes
./ens blend demo/binary      # single vs average vs learned weights; locks one recipe
./ens verdict demo/binary    # opens the holdout once
```

The output of that run is in [examples/demo-binary-record.md](examples/demo-binary-record.md). It ends in NOT SHOWN: learned weights looked 0.009 better in development and 0.006 better on the holdout, with an interval that reaches zero. That is the tool doing its job.

Then open the folder in Claude Code (or another agent) and ask:

> Here are my three churn models. Is it worth ensembling them? Customers repeat across rows, and the probabilities feed a retention budget.

The agent reads [SKILL.md](SKILL.md) and works in this order: the task, metric and `min_gain`; the split; the members as out-of-fold and holdout predictions; check, members, blend; one verdict; then slices, shift and cost. To use the skill from any project: `./ens install`.

For your own models:

```bash
./ens new churn --task binary --split group
# set min_gain and purpose in churn/ensemble.json, write the prediction files, then check, members, blend, verdict
```

Needs Python 3.9 or newer and numpy. Only `demo` and the template need scikit-learn. The tool reads CSV, so the models can come from any library.

## What is in the folder

| Path | What |
|---|---|
| [SKILL.md](SKILL.md) | The entry skill: the twelve rules that do not bend, the loop, the commands |
| [skills/build-ensemble](skills/build-ensemble/SKILL.md) | Eight steps from "what is this for" to the hand-over |
| [skills/review-ensemble](skills/review-ensemble/SKILL.md) | Review someone's ensemble or stacking code, leakage first |
| [docs/rules.md](docs/rules.md) | The 36 rules, with sources and a checklist |
| [docs/file-format.md](docs/file-format.md) | The project folder, `ensemble.json`, and what each command computes |
| [docs/prior-art.md](docs/prior-art.md) | Other people's ensemble skills: what was taken, with credit, and what was not |
| `scripts/ens.py`, `ens` | The tool |
| [templates/make_predictions.py](templates/make_predictions.py) | A scikit-learn script that writes the prediction files correctly |
| `tests/` | `python3 -m unittest discover tests` (numpy only) |

## What it does not do

It does not train or tune your models, fit a non-linear stacker, or serve anything. It cannot tell whether your split matches real use, whether your baseline was tuned fairly, how the result holds on subgroups or shifted data, or whether the gain is worth the serving cost. The skills make the agent check those and say what it did not check.

`./ens lint` matches text patterns in Python files. A hit is a line to read. A clean run is not a pass.

## Licence

MIT. See [LICENSE](LICENSE).
