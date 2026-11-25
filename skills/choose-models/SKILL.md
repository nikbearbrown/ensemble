---
name: choose-models
description: Decides which models to build and try for a prediction problem, starting from a data file and a description of the problem, then builds them inside proper folds. The first half of the ensemble skill; read the root SKILL.md first.
---

# Choose which models to build

The user points you at data and describes the problem. Your job in this skill is to decide which models are worth building, build them so they can be compared fairly, and hand a set of prediction files to `skills/build-ensemble/SKILL.md` for the evidence. Rule numbers refer to `docs/rules.md`.

`./ens suggest` gives you a first list from the shape of the data. It is a starting point written from rules of thumb. You are expected to change it: you can read the problem description, the column names and the domain, and the tool cannot.

## 1. Read the problem, then the data

Get from the user, or from the project, one sentence with the unit predicted, the moment of prediction and what is known at that moment. Then:

```
./ens profile <data file> --target <column> --project <project> --describe "<that sentence>"
        [--id <row id column>] [--group <entity column>] [--time <date column>] [--drop col1,col2]
```

Read every line under "Look at these before building anything". Each is a question to settle now:

| The profile says | Decide |
|---|---|
| A column correlates almost perfectly with the target, or determines it | Is it known at the moment of prediction? If not, it is the answer leaking in: rerun with `--drop`. Ask the user if you cannot tell. |
| An identifier repeats | Will the model meet new customers, patients, sites? Then rerun with `--group`. |
| There is a date column | Is the job to predict the future? Then rerun with `--time`. |
| Small data, or a rare class with few rows per fold | Fewer, simpler members. Say now that a small ensemble gain will not be provable. |
| Exact duplicate rows | Remove them or keep copies on one side of every split. |

The split is the most consequential choice in the whole job (rules 9 and 10), and the tool only guesses at it. Rerun `profile` until the split line is right.

## 2. Get the candidate list, and edit it

```
./ens suggest <project>
```

It writes `PLAN.md` (each model, why, what it brings that the others do not), `candidates.json` (the same list with a `build` flag), and `build_members.py`.

The logic it follows:

- **Tier 1, always:** a regularised linear model and gradient-boosted trees. They differ in kind (additive and smooth against interactions and thresholds), one of them is usually the strongest single model, and the other is the cheapest honest check on it. A text column adds a bag-of-words linear model, because a different representation of the input is the strongest source of real diversity (rule 15).
- **Tier 2, when the data allows:** a random forest, nearest neighbours, a small neural network. Each is a different family. Each is skipped when the data makes it pointless: nearest neighbours on wide or very large data, a neural network on small data.
- **Tier 3, held back:** randomised trees, a kernel machine, a second boosted model with different hyperparameters. Variants of what is already there. Build one only after `./ens blend` shows the first round gained something.

Now use what you know and the tool does not. Edit `candidates.json` and the `CUSTOM` block in `build_members.py`:

- **Domain features beat more models.** A ratio, a lag, a count per entity, an interaction the user's field knows about. A new representation is a new member; the same model on better features often replaces the old one.
- **The right model for the data type.** Forecasts: a naive baseline (last value, same period last season) and lag features built only from the past. Images, audio, long text: a pretrained network's embeddings into a linear model, or a fine-tuned network, as members. Counts or durations: a model with the matching loss.
- **A better boosted library** if it is installed (LightGBM, XGBoost, CatBoost), in place of `boost`, not beside it, unless their errors turn out to differ.
- **Remove what cannot serve.** If the user's latency or memory limit rules a family out, drop it now (rule 4).
- **Do not add seeds of a deterministic model, or five settings of one model.** They are the weakest source of diversity and cost a fit each.

For every member you add, write one line in `PLAN.md`: what it is and what it brings that the others do not. If you cannot write the second half, do not build it.

## 3. Set the bar, then build

Set `min_gain` and `purpose` in `ensemble.json` with the user before anything is trained. `./ens build` refuses to run without `min_gain`.

```
./ens build <project> --list       # what will be trained, how many fits
./ens build <project>
```

`build_members.py` takes the holdout off first, makes one set of folds for the split you chose, and for each member fits a single pipeline (imputing, scaling, encoding and the model together) inside each training fold, predicts the held-out fold, then refits on all development rows to predict the holdout. It prints a floor (predicting the base rate) and each member's out-of-fold score, and records the time each took in `ensemble.json`.

Read the output before going on:

- A member at or below the floor has learned nothing. Find out why before averaging it into anything.
- A member far better than the rest, or near-perfect, is more often a leak than a triumph. Check its features.
- A member that failed is reported and skipped. Fix it or remove it from `candidates.json`.

When you write your own member, the three things the template keeps right are yours to keep right: everything fitted from data is inside the pipeline; the same folds as every other member; the row id in every file.

## 4. Tune the strongest one

Nothing in the template is tuned. Before any comparison means anything, tune the best single model with the effort the ensemble as a whole will get (rule 6), inside the folds: nested, or on an inner split of each training fold. If you tune on the folds themselves, say so in the record: the out-of-fold predictions have then seen their own labels a little (rule 24).

Then tune the other members that are staying, with comparable effort. An ensemble of default models beating one default model shows nothing.

## 5. Hand over to the evidence

```
./ens check <project>
./ens members <project>
./ens blend <project>
```

Continue at step 5 of `skills/build-ensemble/SKILL.md`. `blend` ends with "Members to keep" and "Not needed". That, and the removal table, decide which models stay. Tier 3 candidates and new ideas are built one at a time from here, each with `./ens build <project> --only <name>`, and each must change the blend by `min_gain` to stay.

Changing the member list on development evidence is allowed for as long as the holdout is unopened. After `./ens verdict` it is not.

## What to tell the user

Before building: the split you chose and why, the columns you dropped and why, the list of models with the one-line reason for each, and the number of fits. After building: each member's score against the floor, which were kept, and that nothing is proven until the verdict.
