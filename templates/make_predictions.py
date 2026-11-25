#!/usr/bin/env python3
"""Write the prediction files that ./ens reads. A scikit-learn starting point: edit it for your data.

What this script gets right, and what you must keep right when you change it:

  1. The holdout is split off FIRST and is not looked at again until ./ens verdict.
  2. Every member is a Pipeline, so scaling, imputing and encoding are fitted inside each training fold.
  3. Every member is predicted out-of-fold on the SAME folds.
  4. Each member is then refitted on all development rows to predict the holdout.
  5. Every file carries the row id, so nothing depends on row order.

Not handled here: tuning. If you tune a member on the same folds that make its out-of-fold predictions, those
predictions have seen their own labels a little. With small data, tune inside each fold (nested).

Usage:  python3 make_predictions.py --demo binary|regression --out <project>
        (or delete the demo block and load your own X, y, ids)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default="binary", choices=["binary", "regression"])
ap.add_argument("--out", default=".")
args = ap.parse_args()
out = Path(args.out)
SEED = 42

# ---- data: replace this block with your own X (features), y (target), ids (stable row ids) ----
if args.demo == "binary":
    X, y = make_classification(n_samples=6000, n_features=20, n_informative=6, n_redundant=4,
                               n_clusters_per_class=3, flip_y=0.08, random_state=SEED)
    members = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "boost": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=SEED),
        "boost_seed2": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=7),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stratify = y
else:
    X, y = make_regression(n_samples=4000, n_features=20, n_informative=8, noise=25.0, random_state=SEED)
    y = y + 120 * np.sin(X[:, 0] * 2) + 100 * (X[:, 1] > 0.5) * X[:, 2]   # a part no linear model can fit
    members = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "boost": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=SEED),
        "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    stratify = None
ids = np.array([f"r{i:05d}" for i in range(len(y))])
# ---- end of data block ----

# 1. the holdout comes off first
X_dev, X_hold, y_dev, y_hold, id_dev, id_hold = train_test_split(
    X, y, ids, test_size=0.25, random_state=SEED, stratify=stratify)


def predict(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


write(out / "truth.csv", ["id", "y"], zip(id_dev, y_dev))
write(out / "holdout_truth.csv", ["id", "y"], zip(id_hold, y_hold))

# 3. one set of folds, used by every member
fold_of = np.empty(len(y_dev), dtype=int)
splits = list(folds.split(X_dev, y_dev))
for k, (_, te) in enumerate(splits):
    fold_of[te] = k

for name, model in members.items():
    oof = np.empty(len(y_dev))
    for tr, te in splits:
        m = clone(model).fit(X_dev[tr], y_dev[tr])          # 2. the whole pipeline is fitted on the training fold
        oof[te] = predict(m, X_dev[te])
    write(out / "oof" / f"{name}.csv", ["id", "fold", "pred"], zip(id_dev, fold_of, oof))
    final = clone(model).fit(X_dev, y_dev)                  # 4. refit on all development rows
    write(out / "holdout" / f"{name}.csv", ["id", "pred"], zip(id_hold, predict(final, X_hold)))
    print(f"wrote oof/{name}.csv and holdout/{name}.csv")
