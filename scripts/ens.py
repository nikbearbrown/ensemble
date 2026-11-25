#!/usr/bin/env python3
"""ensemble command. Run ./ens help

Checks whether a machine-learning ensemble earns its place. It works on saved predictions
(CSV files keyed by row id), so it does not care which library trained the models.
Needs numpy. Only `demo` needs scikit-learn.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("numpy was not found. Install it with: python3 -m pip install numpy")
    sys.exit(1)

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
LOWER = {"logloss", "brier", "rmse", "mae"}          # lower is better
METRICS = {"binary": ["logloss", "brier", "auc", "accuracy"],
           "multiclass": ["logloss", "brier", "accuracy"],
           "regression": ["rmse", "mae"]}
EPS = 1e-12


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- files
def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise Fail(f"{path} has no rows")
    return rows


def load_config(proj):
    p = proj / "ensemble.json"
    if not p.exists():
        raise Fail(f"{p} not found. Start a project with: ./ens new {proj}")
    cfg = json.loads(p.read_text())
    task = cfg.get("task")
    if task not in METRICS:
        raise Fail(f'ensemble.json: "task" must be one of {", ".join(METRICS)}')
    cfg.setdefault("metric", METRICS[task][0])
    if cfg["metric"] not in METRICS[task]:
        raise Fail(f'ensemble.json: metric "{cfg["metric"]}" does not fit task "{task}". '
                   f'Use one of {", ".join(METRICS[task])}')
    cfg.setdefault("split", "kfold")
    if cfg["split"] not in ("kfold", "group", "time"):
        raise Fail('ensemble.json: "split" must be kfold, group or time')
    if "min_gain" not in cfg or cfg["min_gain"] is None:
        raise Fail('ensemble.json: set "min_gain", the smallest improvement in the metric that would '
                   "be worth the extra models. Decide it before you look at any ensemble score.")
    cfg["min_gain"] = float(cfg["min_gain"])
    return cfg


def member_names(proj, cfg, sub="oof"):
    names = cfg.get("members") or sorted(p.stem for p in (proj / sub).glob("*.csv"))
    if not names:
        raise Fail(f"No member files in {proj / sub}. See docs/file-format.md")
    return names


def pred_columns(rows, task, path):
    cols = list(rows[0].keys())
    if task == "multiclass":
        pc = [c for c in cols if c.startswith("p_")]
        if len(pc) < 3:
            raise Fail(f"{path}: a multiclass file needs one p_<class> column per class")
        return pc
    if "pred" not in cols:
        raise Fail(f'{path}: needs a "pred" column')
    return ["pred"]


class Data:
    """Truth plus every member's predictions for one layer (oof or holdout), aligned by id."""

    def __init__(self, proj, cfg, layer, members=None):
        self.cfg, self.layer, self.problems, self.warnings = cfg, layer, [], []
        task = cfg["task"]
        tpath = proj / ("truth.csv" if layer == "oof" else "holdout_truth.csv")
        if not tpath.exists():
            raise Fail(f"{tpath} not found. See docs/file-format.md")
        truth = read_csv(tpath)
        for col in ("id", "y"):
            if col not in truth[0]:
                raise Fail(f'{tpath}: needs "id" and "y" columns')
        tids = [r["id"] for r in truth]
        if len(set(tids)) != len(tids):
            self.problems.append(f"{tpath.name}: duplicate ids")
        self.members = members or member_names(proj, cfg, layer)
        tables, self.classes = {}, None
        for m in self.members:
            path = proj / layer / f"{m}.csv"
            if not path.exists():
                raise Fail(f"{path} not found")
            rows = read_csv(path)
            pc = pred_columns(rows, task, path)
            if task == "multiclass":
                if self.classes is None:
                    self.classes = pc
                elif pc != self.classes:
                    self.problems.append(f"{layer}/{m}.csv: class columns {pc} differ in name or order "
                                         f"from {self.classes}")
                    if sorted(pc) != sorted(self.classes):
                        raise Fail(self.problems[-1])
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                self.problems.append(f"{layer}/{m}.csv: duplicate ids")
            if layer == "oof" and "fold" not in rows[0]:
                raise Fail(f'{path}: an out-of-fold file needs a "fold" column')
            tables[m] = {r["id"]: r for r in rows}
        common = set(tables[self.members[0]])
        for m in self.members[1:]:
            other = set(tables[m])
            if other != common:
                self.problems.append(f"{layer}/{m}.csv: ids differ from {self.members[0]}.csv "
                                     f"({len(other - common)} extra, {len(common - other)} missing)")
            common &= other
        missing_truth = common - set(tids)
        if missing_truth:
            self.problems.append(f"{len(missing_truth)} predicted ids have no row in {tpath.name}")
        self.ids = [i for i in tids if i in common]
        if layer == "holdout" and len(self.ids) != len(tids):
            self.problems.append(f"{len(tids) - len(self.ids)} holdout rows have no prediction from every member")
        if not self.ids:
            raise Fail(f"No ids are shared by {tpath.name} and every member file")
        tmap = {r["id"]: r for r in truth}
        ycol = [tmap[i]["y"] for i in self.ids]
        self.group = np.array([tmap[i]["group"] for i in self.ids]) if "group" in truth[0] else None
        self.time = [tmap[i]["time"] for i in self.ids] if "time" in truth[0] else None
        if task == "regression":
            self.y = np.array(ycol, dtype=float)
        elif task == "binary":
            labels = sorted(set(ycol))
            pos = str(cfg.get("positive", labels[-1]))
            if len(labels) != 2 or pos not in labels:
                raise Fail(f"{tpath.name}: a binary task needs exactly two labels in y (found {labels[:6]})")
            self.y = np.array([1.0 if v == pos else 0.0 for v in ycol])
        else:
            names = [c[2:] for c in self.classes]
            bad = sorted(set(ycol) - set(names))
            if bad:
                raise Fail(f"{tpath.name}: labels {bad[:6]} have no p_<class> column")
            self.y = np.array([names.index(v) for v in ycol])
        cols = self.classes or ["pred"]
        stack = []
        for m in self.members:
            t = tables[m]
            try:
                arr = np.array([[float(t[i][c]) for c in cols] for i in self.ids])
            except ValueError:
                raise Fail(f"{layer}/{m}.csv: a prediction is empty or not a number")
            stack.append(arr if task == "multiclass" else arr[:, 0])
        self.S = np.array(stack)                     # (members, rows[, classes])
        for k, m in enumerate(self.members):
            a = self.S[k]
            if not np.isfinite(a).all():
                self.problems.append(f"{layer}/{m}.csv: predictions contain NaN or infinity")
            if task != "regression" and (np.nanmin(a) < 0 or np.nanmax(a) > 1):
                self.problems.append(f"{layer}/{m}.csv: values outside 0 to 1. These are scores, not "
                                     "probabilities. Calibrate the member or convert before averaging")
            if task == "multiclass" and np.abs(a.sum(axis=1) - 1).max() > 1e-3:
                self.problems.append(f"{layer}/{m}.csv: class probabilities do not sum to 1 on every row")
        self.fold = None
        if layer == "oof":
            first = tables[self.members[0]]
            self.fold = np.array([first[i]["fold"] for i in self.ids])
            for m in self.members[1:]:
                if any(tables[m][i]["fold"] != first[i]["fold"] for i in self.ids):
                    self.problems.append(f"oof/{m}.csv: fold assignments differ from {self.members[0]}.csv. "
                                         "Every member must use the same folds")
            self.folds = sorted(set(self.fold), key=lambda v: (0, int(v)) if v.lstrip("-").isdigit() else (1, v))
            self._check_folds()

    def _check_folds(self):
        cfg = self.cfg
        if len(self.folds) < 2:
            self.problems.append("only one fold. Out-of-fold predictions need at least two")
        if cfg["task"] != "regression":
            for f in self.folds:
                if len(set(self.y[self.fold == f].tolist())) < 2:
                    self.warnings.append(f"fold {f} holds only one class, so some metrics cannot be scored on it")
        if cfg["split"] == "group":
            if self.group is None:
                self.problems.append('split is "group" but truth.csv has no group column')
            else:
                seen = {}
                for g, f in zip(self.group, self.fold):
                    seen.setdefault(g, set()).add(f)
                n = sum(1 for v in seen.values() if len(v) > 1)
                if n:
                    self.problems.append(f"{n} groups appear in more than one fold. Their rows leak "
                                         "between training and validation")
        elif self.group is not None:
            self.warnings.append('truth.csv has a group column but split is not "group". If rows from one '
                                 "group can sit in two folds, the scores are too good")
        if cfg["split"] == "time":
            if self.time is None:
                self.warnings.append('split is "time" but truth.csv has no time column, so fold order cannot be checked')
            else:
                def key(v):
                    try:
                        return (0, float(v), "")
                    except ValueError:
                        return (1, 0.0, v)
                last = None
                for f in self.folds:
                    ts = [key(t) for t, ff in zip(self.time, self.fold) if ff == f]
                    if last is not None and min(ts) < last:
                        self.problems.append(f"fold {f} starts before the fold ahead of it ends. Time folds "
                                             "must run forward (rolling-origin backtests)")
                    last = max(ts)


# ----------------------------------------------------------------------------- metrics
def auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(name, task, y, P):
    if len(y) == 0:
        return float("nan")
    if task == "regression":
        e = P - y
        return float(np.sqrt(np.mean(e * e))) if name == "rmse" else float(np.mean(np.abs(e)))
    if task == "binary":
        if name == "logloss":
            q = np.clip(P, EPS, 1 - EPS)
            return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        if name == "brier":
            return float(np.mean((P - y) ** 2))
        if name == "auc":
            return auc(y, P)
        return float(np.mean((P >= 0.5) == (y == 1)))
    rows = np.arange(len(y))
    if name == "logloss":
        return float(-np.mean(np.log(np.clip(P[rows, y], EPS, 1))))
    if name == "brier":
        onehot = np.zeros_like(P)
        onehot[rows, y] = 1
        return float(np.mean(((P - onehot) ** 2).sum(axis=1)))
    return float(np.mean(P.argmax(axis=1) == y))


def gain(name, base, new):
    """Positive means `new` is better than `base`."""
    return base - new if name in LOWER else new - base


def blend(S, w):
    return np.tensordot(np.asarray(w, dtype=float), S, axes=1)


def proper_loss(task, y, P):
    return score("rmse", task, y, P) ** 2 if task == "regression" else score("logloss", task, y, P)


def fit_weights(task, y, S, iters=400):
    """Non-negative weights that sum to one, fitted to log loss (classification) or squared error."""
    M = S.shape[0]
    w = np.full(M, 1.0 / M)
    flat = S.reshape(M, -1)
    n = len(y)
    best = proper_loss(task, y, blend(S, w))
    eta = 1.0
    for _ in range(iters):
        P = blend(S, w)
        if task == "regression":
            d = 2 * (P - y) / n
        elif task == "binary":
            q = np.clip(P, EPS, 1 - EPS)
            d = (q - y) / (q * (1 - q)) / n
        else:
            d = np.zeros_like(P)
            rows = np.arange(n)
            d[rows, y] = -1.0 / np.clip(P[rows, y], EPS, 1) / n
        g = flat @ d.ravel()
        scale = float(np.abs(g - g.min()).max()) or 1.0   # keeps the step free of the target's units
        moved = False
        for _ in range(30):
            cand = w * np.exp(-eta * (g - g.min()) / scale)
            cand /= cand.sum()
            loss = proper_loss(task, y, blend(S, cand))
            if loss <= best:
                moved = not np.allclose(cand, w, atol=1e-9)
                w, best = cand, loss
                eta *= 1.3
                break
            eta /= 2
        if not moved:
            break
    w[w < 1e-4] = 0
    return w / w.sum()


def greedy(task, metric, y, S, max_picks=None):
    """Forward selection with replacement (Caruana et al., 2004). Returns weights and the pick order."""
    M = S.shape[0]
    max_picks = max_picks or 3 * M
    single = [score(metric, task, y, S[k]) for k in range(M)]
    pick = int(np.nanargmin(single) if metric in LOWER else np.nanargmax(single))
    counts, total, cur = np.zeros(M), S[pick].copy(), single[pick]
    counts[pick] = 1
    order = [(pick, cur)]
    while counts.sum() < max_picks:
        n = counts.sum()
        trial = [score(metric, task, y, (total + S[k]) / (n + 1)) for k in range(M)]
        k = int(np.nanargmin(trial) if metric in LOWER else np.nanargmax(trial))
        if not gain(metric, cur, trial[k]) > 1e-9:
            break
        counts[k] += 1
        total += S[k]
        cur = trial[k]
        order.append((k, cur))
    return counts / counts.sum(), order


def best_single(task, metric, y, S):
    s = [score(metric, task, y, S[k]) for k in range(S.shape[0])]
    return int(np.nanargmin(s) if metric in LOWER else np.nanargmax(s))


def onehot_w(M, k):
    w = np.zeros(M)
    w[k] = 1
    return w


FITTERS = {
    "single": lambda t, m, y, S: onehot_w(S.shape[0], best_single(t, m, y, S)),
    "mean": lambda t, m, y, S: np.full(S.shape[0], 1.0 / S.shape[0]),
    "weights": lambda t, m, y, S: fit_weights(t, y, S),
    "greedy": lambda t, m, y, S: greedy(t, m, y, S)[0],
}
LABEL = {"single": "best single model", "mean": "equal-weight average",
         "weights": "learned weights", "greedy": "greedy selection"}


def cross_fit(d, method):
    """Blend each fold with a recipe fitted on the other folds (earlier folds only for time splits)."""
    cfg = d.cfg
    out = np.zeros_like(d.S[0])
    M = d.S.shape[0]
    for i, f in enumerate(d.folds):
        te = d.fold == f
        tr = np.isin(d.fold, d.folds[:i]) if cfg["split"] == "time" else ~te
        if method == "mean" or tr.sum() == 0:
            w = np.full(M, 1.0 / M)
        else:
            w = FITTERS[method](cfg["task"], cfg["metric"], d.y[tr], d.S[:, tr])
        out[te] = blend(d.S[:, te], w)
    return out


def fold_scores(d, P):
    return np.array([score(d.cfg["metric"], d.cfg["task"], d.y[d.fold == f], P[d.fold == f]) for f in d.folds])


# ----------------------------------------------------------------------------- statistics
def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, dd = 1.0, 1 - qab * x / qap
    dd = 1 / (dd if abs(dd) > 1e-30 else 1e-30)
    h = dd
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            dd = 1 + aa * dd
            dd = 1 / (dd if abs(dd) > 1e-30 else 1e-30)
            c = 1 + aa / c
            c = c if abs(c) > 1e-30 else 1e-30
            delta = dd * c
            h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def t_pvalue(t, df):
    return betainc(df / 2, 0.5, df / (df + t * t))


def corrected_t(diffs):
    """Nadeau and Bengio (2003) corrected resampled t-test for k-fold differences."""
    k = len(diffs)
    sd = diffs.std(ddof=1)
    if k < 2 or sd == 0:
        return float("nan"), float("nan")
    t = diffs.mean() / (sd * math.sqrt(1 / k + 1 / (k - 1)))
    return float(t), float(t_pvalue(t, k - 1))


def bootstrap_gain(cfg, y, base, ens, group, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    task, metric = cfg["task"], cfg["metric"]
    if group is not None:
        how = "whole groups"
        _, inv = np.unique(group, return_inverse=True)
        units = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    elif cfg["split"] == "time":
        L = max(2, int(round(math.sqrt(n))))
        how = f"blocks of {L} rows in file order"
        units = [np.arange(s, min(s + L, n)) for s in range(0, n, L)]
    else:
        how, units = "rows", None
    out = []
    for _ in range(n_boot):
        if units is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([units[u] for u in rng.integers(0, len(units), len(units))])
        g = gain(metric, score(metric, task, y[idx], base[idx]), score(metric, task, y[idx], ens[idx]))
        if not math.isnan(g):
            out.append(g)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), how


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = which == b
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
            rows.append((edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return float(total), rows


def residuals(d):
    if d.cfg["task"] == "multiclass":
        rows = np.arange(len(d.y))
        return np.array([1 - d.S[k][rows, d.y] for k in range(len(d.members))])
    return d.y[None, :] - d.S


# ----------------------------------------------------------------------------- output
def fmt(x, nd=5):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(header, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join(lines)


def record(proj, title, body):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    path = proj / "RECORD.md"
    head = "" if path.exists() else ("# Ensemble record\n\nWritten by `ens`. Every `blend` and `verdict` "
                                     "run is added here, including the ones that did not go your way.\n")
    with open(path, "a") as f:
        f.write(f"{head}\n## {title} ({stamp})\n\n```\n{body.rstrip()}\n```\n")


def stop_on_problems(d):
    if d.problems:
        raise Fail(f"{len(d.problems)} problem(s) in the {d.layer} files. Run ./ens check for the list. First:\n"
                   "  " + d.problems[0])


def direction(cfg):
    return "lower is better" if cfg["metric"] in LOWER else "higher is better"


# ----------------------------------------------------------------------------- commands
def cmd_check(a):
    proj = Path(a.project)
    cfg = load_config(proj)
    out, fails, warns = [], 0, 0
    layers = ["oof"] + (["holdout"] if (proj / "holdout_truth.csv").exists() else [])
    loaded = {}
    for layer in layers:
        d = Data(proj, cfg, layer)
        loaded[layer] = d
        out.append(f"{layer}: {len(d.members)} members, {len(d.ids)} aligned rows"
                   + (f", {len(d.folds)} folds" if layer == "oof" else ""))
        for p in d.problems:
            out.append("  FAIL  " + p)
        for w in d.warnings:
            out.append("  WARN  " + w)
        fails += len(d.problems)
        warns += len(d.warnings)
    if "holdout" in loaded:
        o, h = loaded["oof"], loaded["holdout"]
        both = set(o.ids) & set(h.ids)
        if both:
            fails += 1
            out.append(f"  FAIL  {len(both)} ids are in both the development rows and the holdout")
        if o.group is not None and h.group is not None:
            shared = set(o.group.tolist()) & set(h.group.tolist())
            if shared:
                fails += 1
                out.append(f"  FAIL  {len(shared)} groups are in both the development rows and the holdout")
        if set(h.members) != set(o.members):
            warns += 1
            out.append("  WARN  holdout/ and oof/ do not hold the same members")
    else:
        warns += 1
        out.append("  WARN  no holdout_truth.csv. Without an untouched holdout, ./ens verdict cannot run and "
                   "no ensemble score here is independent evidence")
    if cfg["task"] != "regression" and cfg["metric"] == "accuracy":
        warns += 1
        out.append("  WARN  accuracy at a 0.5 threshold is a decision rule, not a probability score. "
                   "Use logloss or brier unless accuracy is truly what the decision needs")
    out.append(f"\nFAIL {fails}  WARN {warns}")
    print("\n".join(out))
    return 1 if fails else 0


def cmd_members(a):
    proj = Path(a.project)
    cfg = load_config(proj)
    d = Data(proj, cfg, "oof")
    stop_on_problems(d)
    task, metric = cfg["task"], cfg["metric"]
    costs = cfg.get("cost", {})
    rows = []
    for k, m in enumerate(d.members):
        fs = fold_scores(d, d.S[k])
        rows.append((m, fmt(score(metric, task, d.y, d.S[k])), fmt(np.nanmean(fs)), fmt(np.nanstd(fs, ddof=1)),
                     costs.get(m, "")))
    rows.sort(key=lambda r: float(r[1]) if r[1] != "nan" else 9e9, reverse=metric not in LOWER)
    print(f"Members on out-of-fold predictions. Metric: {metric} ({direction(cfg)}).\n")
    print(table(["member", "all rows", "fold mean", "fold sd", "cost"], rows))
    R = residuals(d)
    C = np.corrcoef(R)
    M = len(d.members)
    print("\nCorrelation of errors (1.00 means two members make the same mistakes):\n")
    short = [m[:10] for m in d.members]
    print(table([""] + short, [[short[i]] + [f"{C[i, j]:.2f}" for j in range(M)] for i in range(M)]))
    if M > 1:
        rho = float((C.sum() - M) / (M * (M - 1)))
        kept = rho + (1 - rho) / M
        print(f"\nAverage error correlation {rho:.2f}. If the members were equally good, an equal-weight average "
              f"of these {M}\nwould keep about {100 * kept:.0f}% of one member's error variance, and no number of "
              f"similar members\ncould take it below {100 * max(rho, 0):.0f}%. (Ueda and Nakano; an illustration, "
              "exact only for squared error.)")
        if task != "regression":
            wrong = np.array([(d.S[k].argmax(axis=1) != d.y) if task == "multiclass" else ((d.S[k] >= 0.5) != (d.y == 1))
                              for k in range(M)])
            allw = wrong.all(axis=0).mean()
            anyw = wrong.any(axis=0).mean()
            print(f"\nRows every member gets wrong: {100 * allw:.1f}%. Rows at least one gets wrong: {100 * anyw:.1f}%. "
                  "No way of combining\nthese members can fix the first kind (rule 35).")
        pairs = [(d.members[i], d.members[j], C[i, j]) for i in range(M) for j in range(i + 1, M) if C[i, j] > 0.95]
        for x, y_, c in pairs:
            print(f"\nNear-duplicates: {x} and {y_} (error correlation {c:.3f}). Look at the ablation in "
                  "./ens blend before keeping both.")
    return 0


def cmd_blend(a):
    proj = Path(a.project)
    cfg = load_config(proj)
    d = Data(proj, cfg, "oof")
    stop_on_problems(d)
    task, metric, mg = cfg["task"], cfg["metric"], cfg["min_gain"]
    M = len(d.members)
    if M < 2:
        raise Fail("An ensemble needs at least two members")
    out = [f"Project {proj.name}: {task}, metric {metric} ({direction(cfg)}), split {cfg['split']}, "
           f"{M} members, {len(d.ids)} rows, {len(d.folds)} folds.",
           f"Smallest gain worth having (min_gain, set in ensemble.json): {mg}", ""]
    P = {m: cross_fit(d, m) for m in FITTERS}
    F = {m: fold_scores(d, P[m]) for m in FITTERS}
    base = F["single"]
    rows, gains = [], {}
    for m in FITTERS:
        diffs = np.array([gain(metric, b, s) for b, s in zip(base, F[m])])
        gains[m] = float(np.nanmean(diffs))
        if m == "single":
            rows.append((LABEL[m], fmt(np.nanmean(F[m])), "", "", "", ""))
            continue
        wins = int(np.sum(diffs > 0))
        if cfg["split"] == "time":
            p = "n/a"
        else:
            p = fmt(corrected_t(diffs[~np.isnan(diffs)])[1], 3)
        rows.append((LABEL[m], fmt(np.nanmean(F[m])), f"{gains[m]:+.5f}", fmt(np.nanstd(diffs, ddof=1)),
                     f"{wins}/{len(diffs)}", p))
    out.append("Each fold is blended with a recipe fitted on the other folds"
               + (" (earlier folds only)" if cfg["split"] == "time" else "") + ".\n")
    out.append(table(["candidate", "fold mean", "gain vs single", "sd of gain", "folds won", "p (corrected t)"], rows))
    out.append("\nThese are tuning diagnostics, not independent evidence: the base models' own tuning has seen "
               "these rows,\nand you are about to choose a recipe with them. The independent number comes from "
               "./ens verdict."
               + ("\nFolds in a time split are not exchangeable, so no test is run across them."
                  if cfg["split"] == "time" else ""))
    # choose the simplest candidate that clears the bar
    learned = max(("weights", "greedy"), key=lambda m: gains[m])
    chosen, why = "single", f"no candidate beat the best single model by min_gain ({mg})"
    if gains["mean"] >= mg:
        chosen, why = "mean", f"the equal-weight average gained {gains['mean']:+.5f}, at least min_gain"
    if gains[learned] >= mg and gains[learned] - max(gains["mean"], 0.0) >= mg:
        chosen = learned
        why = (f"{LABEL[learned]} gained {gains[learned]:+.5f}, which beats the equal-weight average "
               f"({gains['mean']:+.5f}) by at least min_gain")
    elif chosen == "mean" and gains[learned] > gains["mean"]:
        why += (f". {LABEL[learned].capitalize()} did {gains[learned] - gains['mean']:+.5f} better than that, "
                "less than min_gain, so the simpler recipe stays")
    if a.method:
        chosen, why = a.method, f"forced with --method {a.method}"
    w = FITTERS[chosen](task, metric, d.y, d.S)
    baseline = d.members[best_single(task, metric, d.y, d.S)]
    out.append(f"\nChosen: {LABEL[chosen]}, because {why}.")
    out.append(f"Baseline to beat on the holdout: {baseline} (best single member on all out-of-fold rows).")
    if chosen != "single":
        out.append("\n" + table(["member", "weight"], [(m, f"{x:.3f}") for m, x in zip(d.members, w) if x > 0]))
    # ablation and size curve
    full = score(metric, task, d.y, blend(d.S, np.full(M, 1 / M)))
    ab = []
    for k, m in enumerate(d.members):
        keep = [j for j in range(M) if j != k]
        s = score(metric, task, d.y, d.S[keep].mean(axis=0))
        delta = gain(metric, s, full)                 # positive: the member helps
        ab.append((m, fmt(s), f"{delta:+.5f}", "helps" if delta >= mg else ("hurts" if delta < 0 else "under min_gain")))
    out.append(f"\nMember removal (equal-weight average of all {M}: {fmt(full)}):\n")
    out.append(table(["drop", "score without it", "what it adds", ""], ab))
    _, order = greedy(task, metric, d.y, d.S)
    out.append("\nSize curve (greedy, fitted and scored on the same rows, so optimistic):\n")
    out.append(table(["picks", "added", "score"], [(i + 1, d.members[k], fmt(s)) for i, (k, s) in enumerate(order)]))
    lock = {"method": chosen, "members": d.members, "weights": [round(float(x), 6) for x in w],
            "baseline": baseline, "metric": metric, "min_gain": mg}
    lock["sha"] = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()[:12]
    lock["dev_gain"] = round(gains[chosen], 6)
    (proj / "ensemble.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    out.append(f"\nWrote ensemble.lock.json (recipe {lock['sha']}). ./ens verdict scores exactly this recipe on the holdout.")
    if chosen == "single":
        out.append("On this evidence the ensemble has not earned its place. Ship the single model, or find a member "
                   "that makes different mistakes.")
    text = "\n".join(out)
    print(text)
    record(proj, "blend", text)
    return 0


def cmd_verdict(a):
    proj = Path(a.project)
    cfg = load_config(proj)
    lp = proj / "ensemble.lock.json"
    if not lp.exists():
        raise Fail("No ensemble.lock.json. Run ./ens blend first: the recipe is fixed before the holdout is opened.")
    lock = json.loads(lp.read_text())
    task, metric, mg = cfg["task"], cfg["metric"], cfg["min_gain"]
    if lock["metric"] != metric or lock["min_gain"] != mg:
        raise Fail("The metric or min_gain in ensemble.json changed after the recipe was locked. Run ./ens blend again.")
    d = Data(proj, cfg, "holdout", members=lock["members"])
    stop_on_problems(d)
    ledger_path = proj / "holdout-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    earlier = sorted({e["sha"] for e in ledger if e["sha"] != lock["sha"]})
    out = [f"Holdout: {len(d.ids)} rows. Recipe {lock['sha']}: {LABEL[lock['method']]}. Baseline: {lock['baseline']}."]
    if earlier:
        out.append(f"WARN  This holdout has already scored {len(earlier)} other recipe(s) ({', '.join(earlier)}). "
                   "Choosing between recipes\n      on holdout scores is model selection. This number is no longer "
                   "an untouched estimate. Get fresh data\n      before you claim it (rule 13).")
    base = d.S[d.members.index(lock["baseline"])]
    ens = blend(d.S, lock["weights"])
    sb, se = score(metric, task, d.y, base), score(metric, task, d.y, ens)
    rows = [(f"{lock['baseline']} (single)", fmt(sb)), (f"ensemble ({lock['method']})", fmt(se))]
    extra = [m for m in METRICS[task] if m != metric]
    hdr = ["system", metric] + extra
    if lock["method"] == "single":
        rows = rows[:1]
    rows = [r + tuple(fmt(score(m, task, d.y, p)) for m in extra) for r, p in zip(rows, (base, ens))]
    out.append("\n" + table(hdr, rows))
    result = {"sha": lock["sha"], "date": datetime.date.today().isoformat(), "baseline": sb, "ensemble": se}
    if lock["method"] == "single":
        out.append("\nVERDICT: NO ENSEMBLE. The locked recipe is the single model; the number above is its holdout score.")
        result["verdict"] = "no-ensemble"
    else:
        g = gain(metric, sb, se)
        lo, hi, how = bootstrap_gain(cfg, d.y, base, ens, d.group, a.boot)
        out.append(f"\nGain over the single model: {g:+.5f}  (95% interval {lo:+.5f} to {hi:+.5f}, paired bootstrap "
                   f"over {how}, {a.boot} draws)")
        out.append(f"Gain seen in development: {lock['dev_gain']:+.5f}. Smallest gain worth having: {mg}.")
        if task == "binary":
            eb, _ = ece(d.y, base)
            ee, bins = ece(d.y, ens)
            out.append(f"\nCalibration error (ECE, 10 bins): single {eb:.4f}, ensemble {ee:.4f}. Ensemble by bin:\n")
            out.append(table(["predicted", "rows", "mean predicted", "observed"],
                             [(f"{a_:.1f}-{b_:.1f}", n, f"{mp:.3f}", f"{ob:.3f}") for a_, b_, n, mp, ob in bins]))
            if ee > eb + 0.01:
                out.append("Averaging has cost calibration. If anyone uses the probabilities, recalibrate the "
                           "combined output on data the\nensemble did not train on, and check again (rule 29).")
        costs = cfg.get("cost", {})
        used = [m for m, x in zip(lock["members"], lock["weights"]) if x > 0]
        if costs:
            out.append(f"\nCost as recorded in ensemble.json: single {costs.get(lock['baseline'], '?')}; ensemble members "
                       + ", ".join(f"{m} {costs.get(m, '?')}" for m in used))
        else:
            out.append(f"\nThe ensemble serves {len(used)} models in place of 1. No costs are recorded in ensemble.json; "
                       "measure latency and memory end to end before deciding.")
        if g >= mg and lo > 0:
            v = "KEEP"
            out.append(f"\nVERDICT: KEEP. The gain is at least min_gain and the interval is above zero.")
        elif g > 0 and lo > 0:
            v = "TOO SMALL"
            out.append("\nVERDICT: TOO SMALL. The gain is real but under min_gain. Ship the single model unless the "
                       "bar was wrong, and say so if you move it.")
        else:
            v = "NOT SHOWN"
            out.append("\nVERDICT: NOT SHOWN. The interval includes zero. Ship the single model.")
        result.update({"gain": g, "lo": lo, "hi": hi, "verdict": v})
    out.append("\nNot covered here: slices and subgroups, shifted data, and serving cost. See skills/build-ensemble/SKILL.md step 7.")
    text = "\n".join(out)
    print(text)
    ledger.append(result)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    record(proj, "verdict", text)
    return 0


LINT = [
    (r"cv\s*=\s*[\"']prefit[\"']", "FAIL",
     'cv="prefit": the combiner or calibrator trains on rows the base models have already seen, unless those rows were held back'),
    (r"\.predict(_proba)?\(\s*X_?train\b", "WARN",
     "predictions on the training rows. If these feed a combiner, weights or a calibrator, that is leakage: use out-of-fold predictions (rule 23)"),
    (r"(SMOTE|ADASYN|RandomOverSampler)\s*\(", "WARN",
     "oversampling. It must run inside each training fold (an imblearn Pipeline), never before the split, and never on evaluation rows"),
    (r"final_estimator\s*=\s*\w*(Forest|Boost|XGB|LGBM|CatBoost|MLP)\w*\(", "WARN",
     "a flexible combiner. Start with a regularised linear or logistic model; a flexible one must beat it out of sample (rule 22)"),
    (r"\.(score|predict|predict_proba)\(\s*X_?test\b", "NOTE",
     "the test set is used here. Count the uses: every choice made after looking at a test score spends the holdout (rule 13)"),
    (r"TimeSeriesSplit|GroupKFold|StratifiedGroupKFold", "NOTE",
     "structured split. Check the stacker, the calibrator and the threshold all use the same structure"),
]


def cmd_lint(a):
    target = Path(a.path)
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*.py") if ".venv" not in p.parts)
    if a.ipynb and target.is_dir():
        files += sorted(target.rglob("*.ipynb"))
    hits = 0
    for f in files:
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        text = "\n".join(lines)
        split_at = next((i for i, l in enumerate(lines) if re.search(r"train_test_split\(|\.split\(", l)), None)
        for i, line in enumerate(lines):
            code = line.split("#")[0]
            for pat, level, msg in LINT:
                if re.search(pat, code):
                    print(f"{level}  {f}:{i + 1}  {msg}")
                    hits += 1
            if re.search(r"\.fit_transform\(|\.fit_resample\(", code) and "Pipeline" not in text \
                    and (split_at is None or i < split_at):
                print(f"WARN  {f}:{i + 1}  a transform is fitted before any split and outside a Pipeline. "
                      "Its parameters have seen the evaluation rows (rule 11)")
                hits += 1
        if re.search(r"\bKFold\(|StratifiedKFold\(", text) and re.search(r"date|timestamp|datetime", text, re.I) \
                and "TimeSeriesSplit" not in text:
            print(f"NOTE  {f}  random folds in a file that handles dates. If the task is predicting the future, "
                  "use forward-chaining splits")
            hits += 1
    print(f"\n{len(files)} file(s) read, {hits} line(s) to look at. These are text patterns, not proof: read each line, "
          "and a clean result does not mean no leakage.")
    return 0


def cmd_new(a):
    proj = Path(a.project)
    if proj.exists() and any(proj.iterdir()):
        raise Fail(f"{proj} already exists and is not empty")
    (proj / "oof").mkdir(parents=True)
    (proj / "holdout").mkdir()
    cfg = {"task": a.task, "metric": a.metric or METRICS[a.task][0], "split": a.split, "min_gain": None,
           "cost": {}, "purpose": "What is predicted, for which unit, at what moment, for which population."}
    (proj / "ensemble.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copy(ROOT / "templates" / "make_predictions.py", proj / "make_predictions.py")
    print(f"Made {proj}/ with ensemble.json, oof/, holdout/ and make_predictions.py.\n"
          'Next: set "min_gain" and "purpose" in ensemble.json before training anything, then write the prediction '
          f"files\n(docs/file-format.md; make_predictions.py is a scikit-learn starting point) and run ./ens check {proj}")
    return 0


def cmd_demo(a):
    proj = Path(a.project)
    if proj.exists() and any(proj.iterdir()):
        raise Fail(f"{proj} already exists and is not empty")
    proj.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "templates" / "make_predictions.py", proj / "make_predictions.py")
    cfg = {"task": a.task, "metric": METRICS[a.task][0], "split": "kfold",
           "min_gain": 0.005 if a.task == "binary" else 1.0,
           "cost": {}, "purpose": "Synthetic demo data from scikit-learn. Not a real task."}
    (proj / "ensemble.json").write_text(json.dumps(cfg, indent=2) + "\n")
    r = subprocess.run([sys.executable, str(proj / "make_predictions.py"), "--demo", a.task, "--out", str(proj)])
    if r.returncode:
        raise Fail("The demo needs scikit-learn: python3 -m pip install scikit-learn")
    print(f"\nDemo project in {proj}/. Now run, in order:\n  ./ens check {proj}\n  ./ens members {proj}\n"
          f"  ./ens blend {proj}\n  ./ens verdict {proj}")
    return 0


def cmd_install(a):
    skills = Path(os.environ.get("CLAUDE_SKILLS_DIR", Path.home() / ".claude" / "skills"))
    target = skills / "ensemble"
    if target.exists() or target.is_symlink():
        raise Fail(f"{target} already exists. Remove it yourself first if you want to relink.")
    skills.mkdir(parents=True, exist_ok=True)
    target.symlink_to(ROOT)
    print(f"Linked {target} -> {ROOT}\nStart a new Claude Code session and ask for an ensemble.")
    return 0


HELP = """ensemble %s: does this ensemble earn its place?

  ./ens new <project> --task binary|multiclass|regression [--split kfold|group|time] [--metric m]
  ./ens demo <project> [--task binary|regression]   a worked example (needs scikit-learn)
  ./ens check <project>      are the prediction files aligned, complete, on one scale, leak-free by fold?
  ./ens members <project>    each member alone, and whether they make the same mistakes
  ./ens blend <project>      single vs equal average vs learned weights; picks the simplest that clears
                             min_gain; ablation; size curve; writes ensemble.lock.json
  ./ens verdict <project>    scores the locked recipe once on the holdout: KEEP, TOO SMALL or NOT SHOWN
  ./ens lint <path>          reads Python for common leakage patterns (hints, not proof)
  ./ens install              link this folder into ~/.claude/skills

A project is a folder of CSV files: docs/file-format.md. The rules: docs/rules.md.
""" % VERSION


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ens", add_help=False)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("check", "members"):
        sub.add_parser(name).add_argument("project")
    p = sub.add_parser("blend")
    p.add_argument("project")
    p.add_argument("--method", choices=list(FITTERS))
    p = sub.add_parser("verdict")
    p.add_argument("project")
    p.add_argument("--boot", type=int, default=2000)
    p = sub.add_parser("lint")
    p.add_argument("path")
    p.add_argument("--ipynb", action="store_true")
    p = sub.add_parser("new")
    p.add_argument("project")
    p.add_argument("--task", required=True, choices=list(METRICS))
    p.add_argument("--split", default="kfold", choices=["kfold", "group", "time"])
    p.add_argument("--metric")
    p = sub.add_parser("demo")
    p.add_argument("project")
    p.add_argument("--task", default="binary", choices=["binary", "regression"])
    sub.add_parser("install")
    sub.add_parser("help")
    a = ap.parse_args(argv)
    if a.cmd in (None, "help"):
        print(HELP)
        return 0
    try:
        return globals()["cmd_" + a.cmd](a)
    except Fail as e:
        print(f"ens: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
