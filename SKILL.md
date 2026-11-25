---
name: ensemble
description: Builds, evaluates and reviews machine-learning ensembles (bagging, boosting, voting, blending, stacking, deep ensembles) for classification, regression and forecasting, and decides with evidence whether the ensemble earns its extra cost over the best single model. Use when the user asks to ensemble, blend, stack or average models, combine forecasts, improve a model by combining others, or review or audit ensemble or stacking code for leakage. Works on saved predictions keyed by row id, with a command-line tool (./ens) that checks alignment, measures whether members make different mistakes, compares a single model with simple and learned blends, and scores one locked recipe on an untouched holdout. These are ensembles of predictive models, not groups of AI agents.
---

# ensemble

An ensemble works for one reason: competent models that make different mistakes cancel some of them when combined. Most ensembles that disappoint were not badly combined. They were badly evaluated: the weights, the member list or the stacker were chosen on the same rows that then reported the score. This skill builds the ensemble in an order that keeps the evaluation honest, and measures the result instead of trusting it.

The rules and their sources are in `docs/rules.md`. Read it once before your first ensemble. The numbers below (rule 11, rule 13) are its numbers.

## Which skill

| The user wants | Read |
|---|---|
| An ensemble built, or an existing set of models combined | `skills/build-ensemble/SKILL.md` |
| A review of ensemble or stacking code, a notebook, or a claimed result | `skills/review-ensemble/SKILL.md` |
| To know the file layout the tool reads | `docs/file-format.md` |

## The rules that do not bend

1. **Fix the task, the metric and the smallest gain worth having before any ensemble score exists** (rules 1 to 5). `min_gain` in `ensemble.json` is that number. The tool refuses to run without it. Do not change it after seeing results; if it was wrong, say so in the record.
2. **The baseline is the best single model, tuned with the same effort and scored the same way** (rule 6). An ensemble of under-tuned models beating one under-tuned model shows nothing.
3. **The split matches how the model will be used** (rules 9 to 12). New entities: whole groups held out. The future: rolling-origin backtests, forward only. All preprocessing, resampling, tuning and early stopping happen inside the training side of each split.
4. **Predictions are aligned by row id, never by row order, and sit on one scale** (rules 20 and 21). Probabilities average with probabilities. Margins and scores do not.
5. **Simple first** (rule 19). Best single model, then an equal-weight average, then learned weights or a stacker only if they beat the average by `min_gain`.
6. **Anything learned about the combination is learned from out-of-fold predictions** (rule 23). A stacker or a set of weights fitted on predictions the base models made on their own training rows learns to trust whichever model memorised most.
7. **A score on rows that chose the recipe is a tuning diagnostic.** That includes cross-validating a stacker over a saved prediction table. The independent number comes from a holdout no choice has touched, or from rebuilding the whole pipeline, base-model folds included, inside each outer split (rules 13 and 26).
8. **The holdout is opened once, for one locked recipe** (rule 13). `./ens blend` writes the lock. `./ens verdict` scores it and keeps a ledger; a second recipe on the same holdout is reported as such.
9. **A gain is reported with its uncertainty, on the same rows for every candidate** (rule 28). If the interval includes zero, or the gain is under `min_gain`, the answer is the single model.
10. **Every member is there because removing it costs something** (rules 16 to 18). Different algorithm names are not diversity. Low error correlation alone is not value: a model can be different because it is wrong.
11. **If the probabilities are used, calibration is checked on the combined output** (rule 29). Member disagreement is a signal for review, not a calibrated probability (rule 31).
12. **Nothing is invented.** Report what was run, on what data, with what result. If there is no suitable data or baseline yet, hand over an evaluation plan, not a claimed result.

## The loop

```
purpose, metric, min_gain  ->  split  ->  members, each as out-of-fold + holdout predictions
   ->  ./ens check  ->  ./ens members  ->  ./ens blend (locks one recipe)
   ->  ./ens verdict (opens the holdout once)  ->  slices, shift, cost  ->  keep it or ship the single model
```

`./ens check` reads the files: ids, folds, scale, groups, time order. `./ens members` scores each member alone and shows whether they make the same mistakes. `./ens blend` compares the best single model, the equal-weight average, learned weights and greedy selection, each fold blended by a recipe fitted on the other folds, and picks the simplest that clears `min_gain`. `./ens verdict` gives KEEP, TOO SMALL or NOT SHOWN. Every blend and verdict run is added to `RECORD.md` in the project, including the ones that did not go the user's way. Do not delete entries.

The tool cannot judge five things, and you must: whether the split matches deployment, whether the base models were tuned without seeing their own validation rows, whether the single-model baseline was tuned fairly, how the result holds on the slices and shifts that matter, and whether the gain is worth the serving cost.

## Commands

```
./ens new <project> --task binary|multiclass|regression [--split kfold|group|time]
./ens demo <project>          a worked example on synthetic data (needs scikit-learn)
./ens check <project>
./ens members <project>
./ens blend <project>
./ens verdict <project>
./ens lint <path>             reads Python for leakage patterns: hints to read, not proof
```

Needs Python 3.9 or newer and numpy. The tool reads CSV files, so the models can come from any library. `templates/make_predictions.py` is a scikit-learn starting point that writes the files correctly.

## What this skill does not do

It does not train or tune base models for you beyond the template, serve the ensemble, or fit a non-linear stacker. For a stacker with interactions or original features, use scikit-learn's `StackingClassifier` or `StackingRegressor` (they make the out-of-fold predictions internally through `cv`), save the stack's own out-of-fold and holdout predictions as one more member, and let `./ens` compare it with the single model and the plain average like anything else.
