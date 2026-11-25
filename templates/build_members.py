#!/usr/bin/env python3
"""Train the suggested models and write the prediction files ./ens reads.

Copied into a project by `./ens suggest`. It reads profile.json, candidates.json and ensemble.json from
its own folder. Run it, then edit it: this file is yours.

    python3 build_members.py                 build every candidate marked "build": true
    python3 build_members.py --only knn,mlp  build just these (existing files for other members are kept)
    python3 build_members.py --list          show what would be built

What it gets right, and what you must keep right when you change it:
  1. The holdout comes off FIRST: random rows, whole groups, or the latest period, to match the split.
  2. Every member is one Pipeline, so imputing, scaling and encoding are fitted inside each training fold.
  3. Every member is predicted out-of-fold on the SAME folds, then refitted on all development rows to
     predict the holdout.
  4. Every file carries the row id.

What it does not do: tune anything. All models run near their defaults. Tune the strongest one inside the
folds before you believe a comparison, and say in the record if you tuned on the folds themselves.
"""
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC, SVR

HERE = Path(__file__).resolve().parent
SEED = 42
HOLDOUT = 0.2

ap = argparse.ArgumentParser()
ap.add_argument("--only")
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

prof = json.loads((HERE / "profile.json").read_text())
cfg = json.loads((HERE / "ensemble.json").read_text())
cands = json.loads((HERE / "candidates.json").read_text())
task, split, target, n_folds = prof["task"], cfg["split"], prof["target"], prof.get("folds", 5)
cls = task != "regression"

# ----------------------------------------------------------------------------- data
p = Path(prof["data"])
df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p, sep="\t" if p.suffix in (".tsv", ".tab") else ",")
df = df[df[target].notna()].reset_index(drop=True)
kinds = {c["name"]: c for c in prof["columns"]}
num = [c for c, k in kinds.items() if k["kind"] == "numeric"]
cat = [c for c, k in kinds.items() if k["kind"] == "categorical"]
text = [c for c, k in kinds.items() if k["kind"] == "text"]
dates = [c for c, k in kinds.items() if k["kind"] == "datetime"]

X = pd.DataFrame(index=df.index)
for c in num:
    X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
for c in dates:                      # calendar parts: fixed arithmetic on the row's own date, nothing is fitted
    d = pd.to_datetime(df[c], errors="coerce")
    for part in ("year", "month", "dayofweek", "hour"):
        X[f"{c}__{part}"] = getattr(d.dt, part).astype(float)
        num.append(f"{c}__{part}")
for c in cat:                        # missing becomes its own category: a fixed rule, nothing is fitted
    X[c] = df[c].astype("object").where(df[c].notna(), "missing").astype(str)
if text:
    X["__text__"] = df[text].fillna("").astype(str).agg(" ".join, axis=1)
low_cat = [c for c in cat if not kinds[c].get("high_cardinality")]
high_cat = [c for c in cat if kinds[c].get("high_cardinality")]

y_raw = df[target].to_numpy(dtype=float) if not cls else df[target].astype(str).to_numpy()
ids = df[prof["id"]].astype(str).to_numpy() if prof["id"] else np.array([f"r{i:07d}" for i in range(len(df))])
if len(set(ids)) != len(ids):
    raise SystemExit(f'The id column "{prof["id"]}" has duplicates. Row ids must be unique.')
groups = df[prof["group"]].astype(str).to_numpy() if prof["group"] else None
when = pd.to_datetime(df[prof["time"]]).to_numpy() if prof["time"] else None
labels = sorted(set(y_raw)) if cls else None
y = np.array([labels.index(v) for v in y_raw]) if cls else y_raw       # classes are trained as 0..k-1, in `labels` order
positive = str(cfg.get("positive", labels[-1])) if task == "binary" else None

# ----------------------------------------------------------------------------- 1. the holdout comes off first
idx = np.arange(len(df))
if split == "time":
    order = np.argsort(when, kind="stable")
    cut = int(len(order) * (1 - HOLDOUT))
    dev, hold = order[:cut], order[cut:]                       # both stay in time order
elif split == "group":
    dev, hold = next(GroupShuffleSplit(1, test_size=HOLDOUT, random_state=SEED).split(idx, groups=groups))
else:
    dev, hold = train_test_split(idx, test_size=HOLDOUT, random_state=SEED, stratify=y if cls else None)

# ----------------------------------------------------------------------------- 3. one set of folds for every member
if split == "time":                                            # expanding window: train on the past, predict the next block
    blocks = np.array_split(np.arange(len(dev)), n_folds + 1)
    splits = [(np.concatenate(blocks[:k]), blocks[k]) for k in range(1, n_folds + 1)]
elif split == "group":
    kf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED) if cls else GroupKFold(n_folds)
    splits = list(kf.split(dev, y[dev], groups[dev]))
else:
    kf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED) if cls else KFold(n_folds, shuffle=True, random_state=SEED)
    splits = list(kf.split(dev, y[dev]))
fold_of = np.full(len(dev), -1)
for k, (_, te) in enumerate(splits):
    fold_of[te] = k
has_oof = fold_of >= 0                                         # the first time block has no out-of-fold prediction

# ----------------------------------------------------------------------------- 2. members, each a whole pipeline
def linear_features():
    parts = [("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), low_cat)]
    if high_cat:                     # target encoding is cross-fitted by scikit-learn inside the training fold
        parts.append(("hc", make_pipeline(TargetEncoder(random_state=SEED), StandardScaler()), high_cat))
    return ColumnTransformer(parts)


def tree_features(impute=True):
    parts = [("num", SimpleImputer(strategy="median") if impute else "passthrough", num),
             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)]
    return ColumnTransformer(parts)


def members():
    k = int(min(50, max(5, round(np.sqrt(len(dev)) / 2))))
    if cls:
        m = {
            "linear": make_pipeline(linear_features(), LogisticRegression(C=1.0, max_iter=3000)),
            "boost": make_pipeline(tree_features(False), HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=500, early_stopping=True, random_state=SEED)),
            "forest": make_pipeline(tree_features(), RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            "extra": make_pipeline(tree_features(), ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            "knn": make_pipeline(linear_features(), KNeighborsClassifier(n_neighbors=k)),
            "mlp": make_pipeline(linear_features(), MLPClassifier((128, 64), early_stopping=True, max_iter=300, random_state=SEED)),
            "svm": make_pipeline(linear_features(), SVC(probability=True, random_state=SEED)),
            "boost_slow": make_pipeline(tree_features(False), HistGradientBoostingClassifier(
                learning_rate=0.02, max_iter=1500, max_depth=4, early_stopping=True, random_state=SEED)),
        }
        if text:
            m["text_linear"] = make_pipeline(ColumnTransformer([("t", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True), "__text__")]),
                                             LogisticRegression(C=3.0, max_iter=3000))
    else:
        m = {
            "linear": make_pipeline(linear_features(), Ridge(alpha=1.0)),
            "boost": make_pipeline(tree_features(False), HistGradientBoostingRegressor(
                learning_rate=0.06, max_iter=500, early_stopping=True, random_state=SEED)),
            "forest": make_pipeline(tree_features(), RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            "extra": make_pipeline(tree_features(), ExtraTreesRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
            "knn": make_pipeline(linear_features(), KNeighborsRegressor(n_neighbors=k)),
            "mlp": make_pipeline(linear_features(), MLPRegressor((128, 64), early_stopping=True, max_iter=300, random_state=SEED)),
            "svm": make_pipeline(linear_features(), SVR()),
            "boost_slow": make_pipeline(tree_features(False), HistGradientBoostingRegressor(
                learning_rate=0.02, max_iter=1500, max_depth=4, early_stopping=True, random_state=SEED)),
        }
        if text:
            m["text_linear"] = make_pipeline(ColumnTransformer([("t", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True), "__text__")]),
                                             Ridge(alpha=1.0))
    # ---- CUSTOM: add your own members here. Each must be a full Pipeline (features and model together),
    # ---- and a classifier must have predict_proba. Example:
    # m["boost_lgbm"] = make_pipeline(tree_features(False), lightgbm.LGBMClassifier(n_estimators=400, learning_rate=0.05))
    return m


def predict(model, rows):
    """Probabilities as an (n, classes) array in `labels` order, or a 1-D array for regression."""
    if not cls:
        return model.predict(X.iloc[rows])
    p = model.predict_proba(X.iloc[rows])
    full = np.zeros((len(rows), len(labels)))                  # a training fold can lack a rare class
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = p[:, j]
    return full


def columns(P):
    if task == "regression":
        return ["pred"], P[:, None]
    if task == "binary":
        return ["pred"], P[:, [labels.index(positive)]]
    return ["p_" + c for c in labels], P


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def metric(yt, P):
    if not cls:
        return float(np.sqrt(mean_squared_error(yt, P)))
    return float(log_loss(yt, np.clip(P, 1e-12, 1) / np.clip(P, 1e-12, 1).sum(axis=1, keepdims=True), labels=list(range(len(labels)))))


def run(model):
    """Out-of-fold predictions on the shared folds, then a refit on all development rows for the holdout."""
    oof = np.zeros((len(dev), len(labels))) if cls else np.zeros(len(dev))
    for tr, te in splits:
        oof[te] = predict(clone(model).fit(X.iloc[dev[tr]], y[dev[tr]]), dev[te])
    return oof, predict(clone(model).fit(X.iloc[dev], y[dev]), hold)


available = members()
want = [c["name"] for c in cands if c["build"]]
if args.only:
    want = [w.strip() for w in args.only.split(",")]
unknown = [w for w in want if w not in available]
if unknown:
    raise SystemExit(f"No such member: {', '.join(unknown)}. Available: {', '.join(available)}")
fits = len(splits) + 1
print(f"{len(dev)} development rows ({int(has_oof.sum())} with out-of-fold predictions), {len(hold)} holdout rows, "
      f"{len(splits)} folds, split {split}.")
print(f"Building: {', '.join(want)}  ({fits} fits each)")
if args.list:
    raise SystemExit(0)

extra_cols = (["group"] if groups is not None else []) + (["time"] if when is not None else [])


def truth_rows(rows):
    for r in rows:
        yield [ids[r], y_raw[r]] + ([groups[r]] if groups is not None else []) + ([str(when[r])] if when is not None else [])


write(HERE / "truth.csv", ["id", "y"] + extra_cols, truth_rows(dev))
write(HERE / "holdout_truth.csv", ["id", "y"] + extra_cols, truth_rows(hold))

floor_oof, _ = run(DummyClassifier(strategy="prior") if cls else DummyRegressor())
name_of = "log loss" if cls else "RMSE"
print(f"\nFloor (predict the base rate): {name_of} {metric(y[dev][has_oof], floor_oof[has_oof]):.5f}. "
      "A member that cannot beat this has learned nothing.")
for name in want:
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof, hp = run(available[name])
    except Exception as e:                                     # one member failing must not lose the others
        print(f"  {name:12s} FAILED: {type(e).__name__}: {e}")
        continue
    hdr, O = columns(oof)
    _, H = columns(hp)
    keep = np.flatnonzero(has_oof)
    write(HERE / "oof" / f"{name}.csv", ["id", "fold"] + hdr, ([ids[dev[i]], fold_of[i]] + list(O[i]) for i in keep))
    write(HERE / "holdout" / f"{name}.csv", ["id"] + hdr, ([ids[hold[i]]] + list(H[i]) for i in range(len(hold))))
    secs = time.time() - t0
    cfg.setdefault("cost", {})[name] = f"{secs:.1f} s to train {fits} times and predict"
    print(f"  {name:12s} out-of-fold {name_of} {metric(y[dev][has_oof], oof[has_oof]):.5f}   {secs:.1f} s")
(HERE / "ensemble.json").write_text(json.dumps(cfg, indent=2) + "\n")
try:
    shown = HERE.relative_to(Path.cwd())
except ValueError:
    shown = HERE
print(f"\nNothing here is tuned. Next: ./ens check {shown}, then members, blend, and one verdict.")
