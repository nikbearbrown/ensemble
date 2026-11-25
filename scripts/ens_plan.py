"""profile and suggest: look at a data file, then say which models are worth building and why.

Needs pandas. The suggestions are rules of thumb, explained in skills/choose-models/SKILL.md, not a search:
they pick a first set of models that differ in kind, and say what to skip. Evidence comes later, from
./ens members and ./ens blend.
"""
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIGH_CARD = 50          # more distinct values than this: target-encode for linear models, not one-hot


def _pd():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise RuntimeError("profile needs pandas: python3 -m pip install pandas")


def read_table(path):
    pd = _pd()
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def column_kind(pd, s, n):
    nun = int(s.nunique(dropna=True))
    if nun <= 1:
        return "constant", nun
    if pd.api.types.is_bool_dtype(s):
        return "numeric", nun
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", nun
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and nun == n and n > 50:
            return "id", nun
        return "numeric", nun
    sample = s.dropna().astype(str).head(500)
    if len(sample):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() > 0.9 and not sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
            return "datetime", nun
        if sample.str.len().mean() > 40 and nun > 0.5 * s.notna().sum():
            return "text", nun
    if nun > 0.95 * s.notna().sum() and n > 50:
        return "id", nun
    return "categorical", nun


def infer_task(pd, y):
    nun = y.nunique(dropna=True)
    if nun == 2:
        return "binary"
    if pd.api.types.is_float_dtype(y) or (pd.api.types.is_numeric_dtype(y) and nun > 20):
        return "regression"
    if nun <= 50:
        return "multiclass"
    raise ValueError(f"The target has {nun} distinct non-numeric values. Pass --task to say what this is.")


def profile(data, target, project, id_col=None, group=None, time=None, task=None, describe=None, folds=5, drop=None):
    pd = _pd()
    df = read_table(data)
    for name, col in (("target", target), ("id", id_col), ("group", group), ("time", time)):
        if col and col not in df.columns:
            raise ValueError(f'--{name} "{col}" is not a column. Columns: {", ".join(map(str, df.columns[:30]))}')
    n_all = len(df)
    df = df[df[target].notna()]
    n = len(df)
    y = df[target]
    task = task or infer_task(pd, y)
    dropped = [d.strip() for d in (drop or "").split(",") if d.strip()]
    for d in dropped:
        if d not in df.columns:
            raise ValueError(f'--drop "{d}" is not a column.')
    special = {target, id_col, group, time, *dropped}
    cols, flags = [], []
    if n_all != n:
        flags.append(f"{n_all - n} rows have no target and will be left out.")
    ycode = y if task == "regression" else y.astype("category").cat.codes
    for c in df.columns:
        if c in special:
            continue
        kind, nun = column_kind(pd, df[c], n)
        idname = str(c).lower().endswith("id")
        if idname and kind in ("numeric", "categorical") and nun > 20 and not pd.api.types.is_float_dtype(df[c]):
            kind = "entity"          # an identifier that repeats: not a feature, maybe the unit to hold out
        entry = {"name": str(c), "kind": kind, "unique": nun, "missing": round(float(df[c].isna().mean()), 4)}
        if kind == "categorical" and nun > HIGH_CARD:
            entry["high_cardinality"] = True
        if kind == "numeric" and nun > 2:
            r = df[c].astype(float).corr(ycode.astype(float))
            if r == r and abs(r) > 0.97:
                flags.append(f'"{c}" correlates {r:+.2f} with the target. Is it known at the moment of prediction, '
                             "or is it the answer leaking in?")
        if kind == "categorical" and task != "regression" and 1 < nun <= 0.5 * n:
            if (df.groupby(c, observed=True)[target].nunique() == 1).all() and nun > y.nunique():
                flags.append(f'"{c}" determines the target exactly. Check it is not derived from the target.')
        if kind == "entity" and not group:
            flags.append(f'"{c}" is an identifier that repeats ({nun} values over {n} rows). It is left out of the '
                         f"features. If the model will meet new values of it, hold out whole groups: --group {c}")
        if kind == "datetime" and not time:
            flags.append(f'"{c}" is a date. If the job is to predict the future, split by time: --time {c}')
        if entry["missing"] > 0.5:
            flags.append(f'"{c}" is {100 * entry["missing"]:.0f}% missing.')
        cols.append(entry)
    count = lambda k: sum(1 for c in cols if c["kind"] == k)
    feat_cols = [c["name"] for c in cols if c["kind"] in ("numeric", "categorical", "text", "datetime")]
    dup = int(df.duplicated(subset=feat_cols + [target]).sum()) if feat_cols else 0
    if dup:
        flags.append(f"{dup} rows are exact duplicates of another row. Keep copies on one side of every split.")
    prof = {"data": str(Path(data).resolve()), "target": target, "id": id_col, "group": group, "time": time,
            "rows": n, "task": task, "folds": folds, "dropped": dropped, "columns": cols,
            "n_numeric": count("numeric"), "n_categorical": count("categorical"), "n_text": count("text"),
            "n_datetime": count("datetime"), "duplicates": dup}
    if task != "regression":
        vc = y.astype(str).value_counts()
        prof["classes"] = {k: int(v) for k, v in vc.items()}
        rare = int(vc.min())
        prof["rare_class_rows"] = rare
        per_fold = rare * 0.8 / folds
        if per_fold < 30:
            flags.append(f"The rarest class has {rare} rows, about {per_fold:.0f} per validation fold. Scores will be "
                         "noisy; a small ensemble gain will not be provable.")
        if vc.min() / vc.sum() < 0.1:
            flags.append(f"Imbalanced: the rarest class is {100 * vc.min() / vc.sum():.1f}% of rows. Score with log loss, "
                         "Brier or precision at a recall, not accuracy. Any resampling goes inside the training fold.")
    if group:
        ng = int(df[group].nunique())
        prof["n_groups"] = ng
        if ng < 5 * folds:
            flags.append(f"Only {ng} groups. With {folds} folds each validation fold holds about {ng // folds}; "
                         "treat every score as rough.")
    if n < 1000:
        flags.append(f"Small data ({n} rows). Prefer few, simple members and tune inside the folds; the holdout "
                     f"will be about {n // 5} rows, so expect a wide interval at ./ens verdict.")
    usable = prof["n_numeric"] + prof["n_categorical"] + prof["n_text"] + prof["n_datetime"]
    if usable == 0:
        raise ValueError("No usable feature columns were found.")
    if usable > n / 10:
        flags.append(f"{usable} feature columns for {n} rows. Expect overfitting; regularise, and prefer linear models and forests.")
    prof["flags"] = flags
    proj = Path(project)
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "profile.json").write_text(json.dumps(prof, indent=2) + "\n")
    cfg_path = proj / "ensemble.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    split = "time" if time else ("group" if group else "kfold")
    cfg.update({"task": task, "split": split})
    cfg.setdefault("metric", {"binary": "logloss", "multiclass": "logloss", "regression": "rmse"}[task])
    cfg.setdefault("min_gain", None)
    cfg.setdefault("cost", {})
    if describe or "purpose" not in cfg:
        cfg["purpose"] = describe or "What is predicted, for which unit, at what moment, for which population."
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return prof


def describe_profile(prof):
    lines = [f"{prof['rows']} rows, target \"{prof['target']}\": {prof['task']}."]
    if "classes" in prof:
        top = list(prof["classes"].items())[:6]
        lines.append("Classes: " + ", ".join(f"{k} {v}" for k, v in top) + (" ..." if len(prof["classes"]) > 6 else ""))
    hc = sum(1 for c in prof["columns"] if c.get("high_cardinality"))
    miss = sum(1 for c in prof["columns"] if c["missing"] > 0)
    lines.append(f"Features: {prof['n_numeric']} numeric, {prof['n_categorical']} categorical ({hc} with more than "
                 f"{HIGH_CARD} values), {prof['n_text']} text, {prof['n_datetime']} date. {miss} have missing values.")
    out = [c["name"] for c in prof["columns"] if c["kind"] in ("id", "constant", "entity")] + prof.get("dropped", [])
    if out:
        lines.append("Left out (identifiers, constants, or dropped by you): " + ", ".join(out[:12]))
    split = "time" if prof["time"] else ("group" if prof["group"] else "kfold")
    lines.append(f"Split: {split}" + (f" on {prof['time'] or prof['group']}" if split != "kfold" else
                                      " (random rows; right only if new rows look like old ones)") + ".")
    if prof["flags"]:
        lines.append("\nLook at these before building anything:")
        lines += ["  - " + f for f in prof["flags"]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- suggest
def suggest(prof):
    """Return the candidate list. Tier 1 is built first; tier 2 with it when conditions hold; tier 3 only on evidence."""
    n, task = prof["rows"], prof["task"]
    nnum, ncat, ntext = prof["n_numeric"] + 4 * prof["n_datetime"], prof["n_categorical"], prof["n_text"]
    hc = sum(1 for c in prof["columns"] if c.get("high_cardinality"))
    missing = any(c["missing"] > 0 for c in prof["columns"])
    cls = task != "regression"
    width = nnum + sum(min(c["unique"], HIGH_CARD) for c in prof["columns"] if c["kind"] == "categorical" and not c.get("high_cardinality")) + hc
    fits = prof.get("folds", 5) + 1
    out = []

    def add(name, family, tier, why, differs, build=True, skip=None, cost="low"):
        out.append({"name": name, "family": family, "tier": tier, "build": bool(build and not skip),
                    "why": why, "differs_by": differs, "cost": cost, **({"skipped_because": skip} if skip else {})})

    add("linear", "linear model", 1,
        ("Regularised logistic regression" if cls else "Ridge regression") + " on scaled numbers and one-hot categories"
        + (", with target encoding fitted inside each fold for the high-cardinality columns" if hc else "")
        + ". The floor every other model must beat, and often the member least like the trees.",
        "Additive and smooth: it extrapolates trends that trees flatten, and misses the interactions trees find.")
    add("boost", "gradient-boosted trees", 1,
        "Histogram gradient boosting. On tabular data this is usually the strongest single model, so it is the "
        "likely baseline to tune (rule 6)." + (" Handles the missing values natively." if missing else ""),
        "Sequential trees that chase the remaining error: low bias, finds interactions and thresholds.",
        cost="medium")
    if ntext:
        add("text_linear", "bag of words", 1,
            "TF-IDF of the text column(s) into a linear model. The other members ignore the text entirely.",
            "A different representation of the input, the strongest source of real diversity (rule 15).")
    add("forest", "bagged trees", 2,
        "Random forest: deep trees averaged over bootstrap samples.",
        "Averages variance away where boosting drives bias down. Related to boost (both trees), so check the error "
        "correlation: it earns a place only if removing it costs something.",
        skip="over 500,000 rows: slow for what it adds" if n > 500_000 else None, cost="medium")
    add("knn", "nearest neighbours", 2,
        "k-nearest neighbours on scaled features.",
        "Purely local: no global shape assumed at all. Often weak alone and still useful in a blend.",
        skip=("over 50,000 rows: prediction is too slow" if n > 50_000 else
              "about %d encoded features: distances stop meaning much" % width if width > 60 else
              "no numeric features" if nnum == 0 else None))
    add("mlp", "neural network", 2,
        "A small multilayer perceptron with early stopping on a slice of the training fold.",
        "Smooth non-linear functions of many features at once; fails in different places from trees.",
        skip=("under 2,000 rows: too little data to train it reliably" if n < 2000 else
              "over 500,000 rows with this template; use a proper deep-learning setup" if n > 500_000 else None),
        cost="medium")
    add("extra", "randomised trees", 3,
        "Extremely randomised trees.",
        "Random split points decorrelate it a little from forest and boost. A variant, not a new kind of model.",
        build=False, cost="medium")
    add("svm", "kernel machine", 3,
        "Support vector machine with an RBF kernel" + (", probabilities by internal Platt scaling." if cls else "."),
        "A margin-based smooth boundary. Worth trying on small numeric data.",
        build=False, skip="over 10,000 rows: training time grows with the square of the rows" if n > 10_000 else None,
        cost="high")
    add("boost_slow", "gradient-boosted trees", 3,
        "The boosted model again with a lower learning rate, more rounds and shallower trees.",
        "Hyperparameters only: the weakest source of diversity. Expect error correlation near 1 with boost.",
        build=False, cost="high")
    notes = []
    if n < 1000:
        for c in out:
            if c["name"] in ("knn", "mlp") and c["build"]:
                c["build"], c["skipped_because"] = False, "under 1,000 rows: keep the first round to linear, boost and forest"
    notes.append(f"Each member built costs {fits} fits ({fits - 1} folds and one refit for the holdout).")
    notes.append("Not in the template, worth a member when available: LightGBM, XGBoost or CatBoost in place of (not "
                 "beside) boost; one boosted library is enough unless their errors differ. Seeds add nothing to "
                 "deterministic models.")
    if prof["time"]:
        notes.append("Forecasting: add a naive baseline by hand (last value, or the same period a season ago) and "
                     "lag features built only from the past. The template's time folds are expanding windows with no gap; "
                     "add a gap if labels overlap in time.")
    if ncat == 0 and nnum > 0 and task != "regression":
        notes.append("All-numeric features: linear, knn and svm all depend on scaling, which the template does inside each fold.")
    return out, notes


def write_plan(proj, prof, cands, notes):
    build = [c for c in cands if c["build"]]
    later = [c for c in cands if not c["build"] and "skipped_because" not in c]
    skipped = [c for c in cands if "skipped_because" in c]
    md = ["# Model plan", "", "Written by `./ens suggest`. These are starting suggestions from the shape of the data, not "
          "results. Which models stay is decided by `./ens members` and `./ens blend`.", "", "## The data", "",
          "```", describe_profile(prof), "```", "", "## Build first", ""]
    for c in build:
        md += [f"### {c['name']} ({c['family']}, tier {c['tier']}, cost {c['cost']})", "", c["why"], "",
               f"*What it brings that the others do not:* {c['differs_by']}", ""]
    md += ["## Build only if the first round leaves room", "",
           "Add one at a time, and only after `./ens blend` shows the first round gained something and no member is marked \"hurts\".", ""]
    md += [f"- **{c['name']}** ({c['family']}): {c['why']} {c['differs_by']}" for c in later] + [""]
    if skipped:
        md += ["## Skipped for this data", ""] + [f"- **{c['name']}**: {c['skipped_because']}." for c in skipped] + [""]
    md += ["## Notes", ""] + [f"- {x}" for x in notes] + ["", "## Next", "",
           "1. Set `min_gain` and `purpose` in `ensemble.json`. Nothing runs without `min_gain`.",
           "2. `python3 build_members.py` trains the models marked `\"build\": true` in `candidates.json` and writes the prediction files.",
           "3. Tune the strongest single model inside the folds before trusting any comparison (rule 6).",
           "4. `./ens check`, `./ens members`, `./ens blend`, then one `./ens verdict`.",
           "5. To add your own model, edit the `CUSTOM` block in `build_members.py` and run it with `--only <name>`.", ""]
    (proj / "PLAN.md").write_text("\n".join(md))
    (proj / "candidates.json").write_text(json.dumps(cands, indent=2) + "\n")
    dest = proj / "build_members.py"
    if not dest.exists():
        shutil.copy(ROOT / "templates" / "build_members.py", dest)
