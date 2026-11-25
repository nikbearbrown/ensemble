---
name: build-ensemble
description: Step-by-step procedure for building a machine-learning ensemble and deciding whether to keep it. Part of the ensemble skill; read the root SKILL.md first.
---

# Build an ensemble

If you are starting from a data file and no models exist yet, do `skills/choose-models/SKILL.md` first: it covers steps 1 to 4 here with the tool doing the building, and hands back at step 5. Otherwise work through the steps in order. Each ends with something you can show the user. Rule numbers refer to `docs/rules.md`.

## 1. Look before you build

Read the project: the target, the unit being predicted, the moment of prediction, the data, the current split, the metric and the current best model. Then settle, with the user where it is not written down:

- **The task in one sentence** with the unit, the horizon and the information available at prediction time. "Predict churn" is not enough. Write it in `purpose` in `ensemble.json`.
- **The primary metric**, chosen for the decision the predictions feed. Log loss or Brier score if probabilities are used; a cost or an operating point if a threshold is.
- **`min_gain`**: the smallest improvement in that metric worth more models to train, serve and debug. Ask the user. If they cannot say, propose a number with a reason (for example the fold-to-fold spread of the current model) and get a yes.
- **Limits**: latency, memory, how many models the team can maintain.

If there is no data, no baseline or no way to hold rows out, stop here and write an evaluation plan (split, metric, `min_gain`, candidates, holdout). Do not produce results.

## 2. Choose the split, and take the holdout off first

| The model will meet | Development folds | Holdout |
|---|---|---|
| New rows like the old ones | Stratified k-fold | Random rows, split off first |
| New patients, users, sites, documents | Whole groups per fold (`GroupKFold`) | Whole groups |
| The future | Rolling-origin backtests, forward only, with a gap if labels overlap | The most recent period |

Check for duplicates and near-duplicates (crops of one image, excerpts of one document, repeated measures) and keep them on one side. Then do not look at the holdout again until step 6.

For forecasts, a row gets an out-of-fold prediction only if a model trained strictly before it made one. The earliest rows have none; leave them out of `oof/`. `cross_val_predict` cannot do this (it needs every row held out exactly once), so write the loop.

## 3. Tune the baseline

Tune the strongest single model with the same budget and the same folds the ensemble will get (rule 6). This model is a member and the bar. A random forest or a gradient-boosted model is already an ensemble; that is fine, it is still the bar.

Match the method to the error (rule 7). High variance: bagging. High bias: boosting. Different families failing in different places: blending or stacking. A model that underfits is not helped by bagging it.

## 4. Build members that differ for a reason

Sources of real difference, roughly strongest first: a different model family; a different representation of the features; different training rows or time windows; different feature subsets; different hyperparameters; different seeds (useful for neural networks, close to useless for deterministic or convex models: the demo's two identical boosted models show it).

For each member write two files (`docs/file-format.md`): out-of-fold predictions on the shared folds, and holdout predictions from a refit on all development rows. `templates/make_predictions.py` does this for scikit-learn. Each member is a `Pipeline` so its scaler, imputer, encoder and feature selection are fitted inside the fold. Oversampling goes inside the fold too, and never touches evaluation rows.

If a member is tuned or early-stopped on the same fold it then predicts, its out-of-fold predictions have seen their own labels. Say so in the record, or tune inside the fold.

Classification members must output probabilities. A margin or a raw boosting score is calibrated first, inside the fold, or the member is left out.

## 5. Check, look, blend

```
./ens check <project>      # FAIL 0 before going on
./ens members <project>
./ens blend <project>
```

Read `members` for three things: each member's own score, the pairs that make the same mistakes, and the share of rows every member gets wrong. No combination fixes the last.

Read `blend` in this order: does the equal-weight average clear `min_gain` over the best single model; do learned weights clear the average by `min_gain`; what does removing each member cost. The pruned average does this removal for you and `blend` prints "Members to keep". A member marked "hurts" or "under min_gain" comes out unless there is a stated reason to keep it (a slice it rescues, say), and then `blend` runs again. Changing the member list is selection; it is allowed here, on development rows, and nowhere later.

If `blend` chooses the single model, the honest options are to stop, or to go back to step 4 for a member that is different in kind. More of the same family has not moved it and will not.

These scores are diagnostics. The base models, the member list and the recipe have all been shaped on these rows.

## 6. Open the holdout once

```
./ens verdict <project>
```

- **KEEP**: the gain is at least `min_gain` and its interval is above zero.
- **TOO SMALL**: real, but under the bar the user set.
- **NOT SHOWN**: the interval includes zero.

Anything but KEEP means the single model ships. Do not go back, change the recipe and run `verdict` again to see if it passes: the ledger will show it, and the second number is not an untouched estimate. If the recipe must change after the holdout was opened, say so and get fresh rows to confirm.

## 7. What the tool does not cover

Before recommending KEEP, check and report:

- **Slices**: the metric by class, subgroup, site and input quality, with row counts. Look at confident mistakes and at rows every member gets wrong.
- **Shift**: a later period, a new site, changed missingness, a feature all members share going bad. Calibration measured on the old distribution does not carry over.
- **Calibration**, if probabilities drive decisions. If the combined output is recalibrated, the calibrator is fitted on predictions from an ensemble that did not train on those rows, and checked on others (rule 30).
- **Thresholds** are chosen on development rows and frozen before the holdout.
- **Cost**: end-to-end latency, memory, and the retraining and monitoring load. Put measured costs in `cost` in `ensemble.json` so the verdict prints them. If the gain is worth having but the cost is not, distil the ensemble into one model and evaluate that model as a new system.
- **Failure**: what happens when one member times out or loses a feature. Dropping it and renormalising the weights is a different model and needs its own evaluation.

## 8. Hand over

Say, in this order: the verdict and the number with its interval; what was actually run and on what data; what was not checked; what you changed in the project; and whether the ensemble earned its place. Point to `RECORD.md`. Version the members, the folds, the out-of-fold predictions, the lock and the seeds together.
