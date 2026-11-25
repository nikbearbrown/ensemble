# Ensemble record

Written by `ens`. Every `blend` and `verdict` run is added here, including the ones that did not go your way.

## blend (2026-11-25 12:30)

```
Project binary: binary, metric logloss (lower is better), split kfold, 5 members, 4500 rows, 5 folds.
Smallest gain worth having (min_gain, set in ensemble.json): 0.005

Each fold is blended with a recipe fitted on the other folds.

candidate             fold mean  gain vs single  sd of gain  folds won  p (corrected t)
--------------------  ---------  --------------  ----------  ---------  ---------------
best single model     0.38718
equal-weight average  0.41068    -0.02349        0.00838     0/5        0.014
learned weights       0.37778    +0.00940        0.00431     5/5        0.031
greedy selection      0.37803    +0.00916        0.00447     5/5        0.038

These are tuning diagnostics, not independent evidence: the base models' own tuning has seen these rows,
and you are about to choose a recipe with them. The independent number comes from ./ens verdict.

Chosen: learned weights, because learned weights gained +0.00940, which beats the equal-weight average (-0.02349) by at least min_gain.
Baseline to beat on the holdout: boost (best single member on all out-of-fold rows).

member       weight
-----------  ------
boost        0.358
boost_seed2  0.358
forest       0.284

Member removal (equal-weight average of all 5: 0.41068):

drop         score without it  what it adds
-----------  ----------------  ------------  --------------
boost        0.42716           +0.01649      helps
boost_seed2  0.42716           +0.01649      helps
forest       0.41549           +0.00481      under min_gain
knn          0.40363           -0.00704      hurts
logreg       0.38883           -0.02185      hurts

Size curve (greedy, fitted and scored on the same rows, so optimistic):

picks  added   score
-----  ------  -------
1      boost   0.38718
2      forest  0.38053
3      boost   0.37775
4      boost   0.37767

Wrote ensemble.lock.json (recipe 39b91ba5274d). ./ens verdict scores exactly this recipe on the holdout.
```

## verdict (2026-11-25 12:30)

```
Holdout: 1500 rows. Recipe 39b91ba5274d: learned weights. Baseline: boost.

system              logloss  brier    auc      accuracy
------------------  -------  -------  -------  --------
boost (single)      0.38652  0.11981  0.90800  0.82667
ensemble (weights)  0.38061  0.11791  0.90987  0.83200

Gain over the single model: +0.00591  (95% interval -0.00002 to +0.01247, paired bootstrap over rows, 2000 draws)
Gain seen in development: +0.00940. Smallest gain worth having: 0.005.

Calibration error (ECE, 10 bins): single 0.0316, ensemble 0.0279. Ensemble by bin:

predicted  rows  mean predicted  observed
---------  ----  --------------  --------
0.0-0.1    301   0.057           0.066
0.1-0.2    195   0.144           0.113
0.2-0.3    115   0.248           0.200
0.3-0.4    68    0.355           0.324
0.4-0.5    85    0.449           0.553
0.5-0.6    80    0.550           0.550
0.6-0.7    78    0.651           0.603
0.7-0.8    67    0.750           0.806
0.8-0.9    186   0.858           0.871
0.9-1.0    325   0.937           0.957

The ensemble serves 3 models in place of 1. No costs are recorded in ensemble.json; measure latency and memory end to end before deciding.

VERDICT: NOT SHOWN. The interval includes zero. Ship the single model.

Not covered here: slices and subgroups, shifted data, and serving cost. See skills/build-ensemble/SKILL.md step 7.
```
