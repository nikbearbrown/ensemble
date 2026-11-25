# ensemble

Point it at a data file and describe the problem. It suggests which machine-learning models to build, builds them, and tells you with evidence which to keep and whether combining them beats the best single one. It is a set of skills for Claude, Codex or any coding agent, and a command-line tool, `./ens`.

These are ensembles of predictive models (bagging, boosting, blending, stacking, deep ensembles), not groups of AI agents.

## In short

- `./ens profile` reads your data: the task, the feature types, and warnings about leaks and about how to split. `./ens suggest` proposes a first set of models that differ in kind, says why each is there, and says what to skip. `./ens build` trains them inside proper folds.
- The suggestions are rules of thumb from the shape of the data. The agent, or you, edits the list with what the tool cannot know: the domain, better features, limits on serving.
- An ensemble helps only when competent models make different mistakes. Most that disappoint were not badly combined; they were badly evaluated.
- Each model's predictions are saved as CSV files keyed by row id (`build` writes them; models from any other library can be added by hand). `./ens` checks them, shows whether the models make the same mistakes, compares the best single model with a plain average, a pruned average and learned weights, says which members to keep, locks one recipe, and scores it once on a holdout nothing has touched.
- The answer is one of three words: **KEEP**, **TOO SMALL** or **NOT SHOWN**. Anything but KEEP means ship the single model.
- You decide the smallest gain worth having (`min_gain`) before any ensemble score exists. The tool will not run without it.

## Get started

```bash
git clone https://github.com/nikbearbrown/ensemble
cd ensemble
./ens help
```

Start from your own data (needs pandas and scikit-learn):

```bash
./ens profile churn.csv --target churned --project churn --group customer_id \
      --describe "Whether a customer cancels within 30 days, from what is known on day 0"
./ens suggest churn          # which models to build, and why: churn/PLAN.md
# set "min_gain" in churn/ensemble.json: the smallest gain that would be worth extra models
./ens build churn            # trains them and writes their predictions
./ens check churn
./ens members churn
./ens blend churn            # which members to keep; locks one recipe
./ens verdict churn          # opens the holdout once
```

[examples/churn-PLAN.md](examples/churn-PLAN.md) is the plan `suggest` wrote for a synthetic churn file.

Or see the evaluation half alone on synthetic predictions:

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

The agent reads [SKILL.md](SKILL.md) and works in this order: the problem and the data; the split; which models to build and why; `min_gain`; build and tune; check, members, blend; one verdict; then slices, shift and cost. To use the skill from any project: `./ens install`.

If you already have trained models and only want them judged:

```bash
./ens new churn --task binary --split group
# set min_gain and purpose in churn/ensemble.json, write the prediction files, then check, members, blend, verdict
```

Needs Python 3.9 or newer and numpy. `profile`, `suggest`, `build` and `demo` also need pandas and scikit-learn. The evaluation commands read CSV, so models from any library can be added.

## What is in the folder

| Path | What |
|---|---|
| [SKILL.md](SKILL.md) | The entry skill: the twelve rules that do not bend, the loop, the commands |
| [skills/choose-models](skills/choose-models/SKILL.md) | From a data file and a problem to a built set of models: split, candidates, what to add by hand |
| [skills/build-ensemble](skills/build-ensemble/SKILL.md) | Eight steps from "what is this for" to the hand-over |
| [skills/review-ensemble](skills/review-ensemble/SKILL.md) | Review someone's ensemble or stacking code, leakage first |
| [docs/rules.md](docs/rules.md) | The 36 rules, with sources and a checklist |
| [docs/file-format.md](docs/file-format.md) | The project folder, `ensemble.json`, and what each command computes |
| [docs/prior-art.md](docs/prior-art.md) | Other people's ensemble skills: what was taken, with credit, and what was not |
| `scripts/ens.py`, `scripts/ens_plan.py`, `ens` | The tool |
| [templates/build_members.py](templates/build_members.py) | The script `suggest` copies into your project: trains the members inside the folds. Yours to edit |
| [templates/make_predictions.py](templates/make_predictions.py) | A smaller scikit-learn example, used by `demo` |
| `tests/` | `python3 -m unittest discover tests` |

## What it does not do

It does not tune models: `build` runs them near their defaults, and tuning the strongest one before trusting a comparison is a step the skill gives the agent. It does not engineer features, fit a non-linear stacker, or serve anything. The suggestions are a starting list, not a search over models. It cannot tell whether a flagged column is really a leak, whether your split matches real use, whether your baseline was tuned fairly, how the result holds on subgroups or shifted data, or whether the gain is worth the serving cost. The skills make the agent check those and say what it did not check.

`./ens lint` matches text patterns in Python files. A hit is a line to read. A clean run is not a pass.

## Licence

MIT. See [LICENSE](LICENSE).
