# The project folder

`./ens` reads a folder of CSV files. Any library can write them. Ids are compared as text, so nothing depends on row order.

```
my-project/
  profile.json             written by ./ens profile: what the data looks like
  candidates.json          written by ./ens suggest: the models, each with a "build" flag you can edit
  PLAN.md                  written by ./ens suggest: each model, why, and what it brings
  build_members.py         copied by ./ens suggest: trains the members and writes the files below; yours to edit
  ensemble.json            the task, the metric, the split, min_gain
  truth.csv                the development rows: id, y [, group] [, time]
  holdout_truth.csv        the holdout rows:     id, y [, group]
  oof/<member>.csv         out-of-fold predictions: id, fold, pred
  holdout/<member>.csv     holdout predictions:     id, pred
  ensemble.lock.json       written by ./ens blend: the one recipe the holdout will score
  holdout-ledger.json      written by ./ens verdict: every time the holdout was scored
  RECORD.md                written by blend and verdict: the full output of every run
```

## ensemble.json

```json
{
  "task": "binary",
  "metric": "logloss",
  "split": "kfold",
  "min_gain": 0.005,
  "purpose": "Whether an active customer cancels within 30 days, from what is known on day 0.",
  "cost": {"boost": "4 ms", "forest": "11 ms"}
}
```

| Key | Values |
|---|---|
| `task` | `binary`, `multiclass`, `regression` |
| `metric` | binary: `logloss`, `brier`, `auc`, `accuracy`. multiclass: `logloss`, `brier`, `accuracy`. regression: `rmse`, `mae`. Default: the first. |
| `split` | `kfold`, `group` (needs a `group` column in `truth.csv`), `time` (folds are forecast origins, in order) |
| `min_gain` | Required. The smallest improvement in the metric worth the extra models, in the metric's own units. |
| `positive` | Optional, binary only: the label that `pred` is the probability of. Default: the label that sorts last (so `1` for 0/1). |
| `members` | Optional list of member names. Default: every file in `oof/`. |
| `cost` | Optional. Any text per member (latency, memory, dollars). Printed with the verdict. |
| `purpose` | The task in one sentence. For people; the tool does not read it. |

## Prediction files

- **Binary**: `pred` is the probability of the positive label, between 0 and 1.
- **Regression**: `pred` is the predicted value, in the target's own units (undo any log or scaling first).
- **Multiclass**: one column per class named `p_<label>`, the same names in the same order in every file, each row summing to 1. `y` in the truth files holds the label.

`fold` in an `oof/` file is the fold in which that row was held out. Every member must give the same row the same fold.

A row in `oof/` means: a model that did not train on this row predicted it. For `time` splits that means a model trained only on earlier data, so the earliest rows have no out-of-fold prediction and are left out of `oof/`. Fold labels must sort in time order (0, 1, 2, ...); with a `time` column in `truth.csv` the tool checks that no fold starts before the one ahead of it ends. For time splits the holdout file should be in time order, because the bootstrap resamples blocks of consecutive rows.

A row in `holdout/` is predicted by the member refitted on all development rows.

## What the commands compute

- **profile**: reads a CSV, TSV or Parquet file. Infers the task from the target, sorts columns into numeric, categorical (flagging those with more than 50 values), text, date, row id, repeating identifier and constant, and reports missing values. Warns about columns that track the target almost exactly, identifiers that repeat (a reason to split by group), date columns (a reason to split by time), duplicates, small data and rare classes. Writes `profile.json` and creates or updates `ensemble.json`.
- **suggest**: turns the profile into a list of candidate models in three tiers (see `skills/choose-models/SKILL.md`), each with a reason, what it brings that the others do not, and a reason when it is skipped. Rules of thumb, not a search.
- **build**: runs the project's `build_members.py`. Holdout first (20%: random rows, whole groups, or the latest period). One set of folds: stratified, by group, or expanding windows in time order. Each member is one pipeline fitted per training fold, then refitted on all development rows for the holdout. Linear-style members get median imputing with missing indicators, scaling, one-hot encoding, and target encoding (cross-fitted inside the fold) for categorical columns with more than 50 values. Tree-style members get ordinal-encoded categories. Dates become year, month, weekday and hour.

- **check**: ids match across members and truth, no duplicates, no missing values, probabilities in range, shared folds, no group in two folds, time folds in order, no id or group in both development and holdout.
- **members**: each member's score on all out-of-fold rows and across folds; the correlation of errors (truth minus prediction; for multiclass, one minus the probability given to the true class); the share of rows every member gets wrong.
- **blend**: five candidates. *Best single model*: chosen on the other folds. *Equal-weight average of all*. *Pruned average*: starting from all members, the member whose removal leaves the best score is removed for as long as some removal costs less than `min_gain`; the rest are averaged equally. *Learned weights*: non-negative, summing to one, fitted to log loss (classification) or squared error (regression). *Greedy selection*: forward selection with replacement on the primary metric (Caruana et al., 2004). For each fold the recipe is fitted on the other folds (earlier folds only for `time`) and scored on that fold. The p-value is Nadeau and Bengio's corrected resampled t-test on the per-fold gains; it is not run for time splits. The chosen recipe climbs a ladder: the average of all members if it gains at least `min_gain` over the single model; the pruned average if it beats that by a further `min_gain`; learned weights or greedy selection if they beat that by a further `min_gain`; otherwise the single model. The lock stores that recipe refitted on all out-of-fold rows.
- **verdict**: the locked recipe and the best single member, both scored on the same holdout rows. The interval is a paired bootstrap of the gain: over rows, over whole groups if `holdout_truth.csv` has a `group` column, or over blocks of consecutive rows for time splits. KEEP needs the gain at or above `min_gain` and the lower end of the interval above zero.
