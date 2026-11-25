# Model plan

Written by `./ens suggest`. These are starting suggestions from the shape of the data, not results. Which models stay is decided by `./ens members` and `./ens blend`.

## The data

```
6000 rows, target "churned": binary.
Classes: 0 4263, 1 1737
Features: 3 numeric, 2 categorical (1 with more than 50 values), 0 text, 1 date. 2 have missing values.
Left out (identifiers, constants, or dropped by you): refund_flag
Split: group on customer_id.

Look at these before building anything:
  - "signup_day" is a date. If the job is to predict the future, split by time: --time signup_day
```

## Build first

### linear (linear model, tier 1, cost low)

Regularised logistic regression on scaled numbers and one-hot categories, with target encoding fitted inside each fold for the high-cardinality columns. The floor every other model must beat, and often the member least like the trees.

*What it brings that the others do not:* Additive and smooth: it extrapolates trends that trees flatten, and misses the interactions trees find.

### boost (gradient-boosted trees, tier 1, cost medium)

Histogram gradient boosting. On tabular data this is usually the strongest single model, so it is the likely baseline to tune (rule 6). Handles the missing values natively.

*What it brings that the others do not:* Sequential trees that chase the remaining error: low bias, finds interactions and thresholds.

### forest (bagged trees, tier 2, cost medium)

Random forest: deep trees averaged over bootstrap samples.

*What it brings that the others do not:* Averages variance away where boosting drives bias down. Related to boost (both trees), so check the error correlation: it earns a place only if removing it costs something.

### knn (nearest neighbours, tier 2, cost low)

k-nearest neighbours on scaled features.

*What it brings that the others do not:* Purely local: no global shape assumed at all. Often weak alone and still useful in a blend.

### mlp (neural network, tier 2, cost medium)

A small multilayer perceptron with early stopping on a slice of the training fold.

*What it brings that the others do not:* Smooth non-linear functions of many features at once; fails in different places from trees.

## Build only if the first round leaves room

Add one at a time, and only after `./ens blend` shows the first round gained something and no member is marked "hurts".

- **extra** (randomised trees): Extremely randomised trees. Random split points decorrelate it a little from forest and boost. A variant, not a new kind of model.
- **svm** (kernel machine): Support vector machine with an RBF kernel, probabilities by internal Platt scaling. A margin-based smooth boundary. Worth trying on small numeric data.
- **boost_slow** (gradient-boosted trees): The boosted model again with a lower learning rate, more rounds and shallower trees. Hyperparameters only: the weakest source of diversity. Expect error correlation near 1 with boost.

## Notes

- Each member built costs 6 fits (5 folds and one refit for the holdout).
- Not in the template, worth a member when available: LightGBM, XGBoost or CatBoost in place of (not beside) boost; one boosted library is enough unless their errors differ. Seeds add nothing to deterministic models.

## Next

1. Set `min_gain` and `purpose` in `ensemble.json`. Nothing runs without `min_gain`.
2. `python3 build_members.py` trains the models marked `"build": true` in `candidates.json` and writes the prediction files.
3. Tune the strongest single model inside the folds before trusting any comparison (rule 6).
4. `./ens check`, `./ens members`, `./ens blend`, then one `./ens verdict`.
5. To add your own model, edit the `CUSTOM` block in `build_members.py` and run it with `--only <name>`.
