"""Tests for ./ens. numpy only: predictions are simulated, no models are trained.
Run: python3 -m unittest discover tests
"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ens  # noqa: E402


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make(tmp, kind="diverse", task="binary", split="kfold", n=3000, min_gain=0.003, seed=0):
    """A project whose members are the truth plus noise. kind: diverse (independent noise),
    clones (shared noise), or weak (one good member, the rest near chance)."""
    rng = np.random.default_rng(seed)
    proj = Path(tmp) / "p"
    z = rng.normal(size=2 * n)
    shared = rng.normal(size=2 * n)
    names = ["a", "b", "c", "d"]
    members = {}
    for k, m in enumerate(names):
        if kind == "diverse":
            noise = rng.normal(size=2 * n)
        elif kind == "clones":
            noise = shared + 0.02 * rng.normal(size=2 * n)
        else:
            noise = rng.normal(size=2 * n) * (0.5 if k == 0 else 6.0)
        members[m] = z + noise
    if task == "binary":
        y = (z + 0.5 * rng.normal(size=2 * n) > 0).astype(int)
        members = {m: 1 / (1 + np.exp(-1.5 * v)) for m, v in members.items()}
    else:
        y = z * 10
        members = {m: v * 10 for m, v in members.items()}
    ids = [f"r{i}" for i in range(2 * n)]
    group = [f"g{i // 10}" for i in range(2 * n)]
    fold = [(i // 10) % 5 if split != "time" else min(4, i * 5 // n) for i in range(n)]
    write(proj / "truth.csv", ["id", "y", "group", "time"], [(ids[i], y[i], group[i], i) for i in range(n)])
    write(proj / "holdout_truth.csv", ["id", "y", "group"], [(ids[i], y[i], group[i]) for i in range(n, 2 * n)])
    for m, v in members.items():
        write(proj / "oof" / f"{m}.csv", ["id", "fold", "pred"], [(ids[i], fold[i], v[i]) for i in range(n)])
        write(proj / "holdout" / f"{m}.csv", ["id", "pred"], [(ids[i], v[i]) for i in range(n, 2 * n)])
    (proj / "ensemble.json").write_text(json.dumps(
        {"task": task, "metric": ens.METRICS[task][0], "split": split, "min_gain": min_gain}))
    return proj


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ens.main(list(argv))
    return code, buf.getvalue()


class Maths(unittest.TestCase):
    def test_auc(self):
        y = np.array([0, 0, 1, 1.0])
        self.assertAlmostEqual(ens.auc(y, np.array([0.1, 0.4, 0.35, 0.8])), 0.75)
        self.assertAlmostEqual(ens.auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)

    def test_logloss_and_rmse(self):
        y = np.array([1.0, 0.0])
        self.assertAlmostEqual(ens.score("logloss", "binary", y, np.array([0.8, 0.1])),
                               -(np.log(0.8) + np.log(0.9)) / 2)
        self.assertAlmostEqual(ens.score("rmse", "regression", np.array([0.0, 0.0]), np.array([3.0, 4.0])),
                               np.sqrt(12.5))

    def test_t_pvalue(self):
        self.assertAlmostEqual(ens.t_pvalue(2.776, 4), 0.05, places=3)   # t table, 4 df, two-sided
        self.assertAlmostEqual(ens.t_pvalue(0.0, 4), 1.0)

    def test_weights_are_a_valid_mixture_and_find_the_good_member(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=2000)
        S = np.array([y + 0.1 * rng.normal(size=2000), y + 3 * rng.normal(size=2000), rng.normal(size=2000)])
        w = ens.fit_weights("regression", y, S)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue((w >= 0).all())
        self.assertGreater(w[0], 0.95)

    def test_weights_split_between_equally_good_independent_members(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=5000)
        S = np.array([y + rng.normal(size=5000) for _ in range(4)])
        w = ens.fit_weights("regression", y, S)
        self.assertTrue(np.allclose(w, 0.25, atol=0.05), w)

    def test_gain_direction(self):
        self.assertGreater(ens.gain("logloss", 0.5, 0.4), 0)
        self.assertGreater(ens.gain("auc", 0.8, 0.9), 0)


class Projects(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def lock(self, proj):
        return json.loads((proj / "ensemble.lock.json").read_text())

    def test_diverse_members_keep_the_equal_average(self):
        proj = make(self.tmp.name, "diverse")
        self.assertEqual(run("check", str(proj))[0], 0)
        self.assertEqual(run("blend", str(proj))[0], 0)
        self.assertEqual(self.lock(proj)["method"], "mean")
        code, out = run("verdict", str(proj), "--boot", "300")
        self.assertIn("VERDICT: KEEP", out)

    def test_clones_do_not_earn_an_ensemble(self):
        proj = make(self.tmp.name, "clones")
        code, out = run("members", str(proj))
        self.assertIn("Near-duplicates", out)
        run("blend", str(proj))
        self.assertEqual(self.lock(proj)["method"], "single")
        self.assertIn("NO ENSEMBLE", run("verdict", str(proj))[1])

    def test_weak_members_do_not_drag_the_choice(self):
        proj = make(self.tmp.name, "weak")
        run("blend", str(proj))
        self.assertEqual(self.lock(proj)["method"], "single")

    def test_regression_and_group_split(self):
        proj = make(self.tmp.name, "diverse", task="regression", split="group", min_gain=0.5)
        self.assertEqual(run("check", str(proj))[0], 0)
        run("blend", str(proj))
        code, out = run("verdict", str(proj), "--boot", "300")
        self.assertIn("whole groups", out)
        self.assertIn("VERDICT: KEEP", out)

    def test_time_split_fits_on_earlier_folds_only(self):
        proj = make(self.tmp.name, "diverse", split="time")
        self.assertEqual(run("check", str(proj))[0], 0)
        code, out = run("blend", str(proj))
        self.assertIn("earlier folds only", out)
        self.assertIn("n/a", out)

    def test_check_catches_misaligned_ids_folds_scale_and_leaks(self):
        proj = make(self.tmp.name, "diverse", split="group")
        rows = list(csv.reader((proj / "oof" / "b.csv").read_text().splitlines()))
        rows[1][1] = "3" if rows[1][1] != "3" else "2"          # one row moved to another fold
        rows[2][2] = "1.7"                                       # a score, not a probability
        rows.pop()                                               # one id missing
        write(proj / "oof" / "b.csv", rows[0], rows[1:])
        t = list(csv.reader((proj / "holdout_truth.csv").read_text().splitlines()))
        d = list(csv.reader((proj / "truth.csv").read_text().splitlines()))
        t[1][2] = d[1][2]                                        # a development group in the holdout
        write(proj / "holdout_truth.csv", t[0], t[1:])
        code, out = run("check", str(proj))
        self.assertEqual(code, 1)
        for needle in ("ids differ", "fold assignments differ", "values outside 0 to 1",
                       "groups are in both"):
            self.assertIn(needle, out)
        self.assertEqual(run("blend", str(proj))[0], 1)          # blend refuses

    def test_min_gain_is_required(self):
        proj = make(self.tmp.name, "diverse")
        cfg = json.loads((proj / "ensemble.json").read_text())
        cfg["min_gain"] = None
        (proj / "ensemble.json").write_text(json.dumps(cfg))
        code, out = run("blend", str(proj))
        self.assertEqual(code, 1)
        self.assertIn("min_gain", out)

    def test_verdict_needs_a_lock_and_the_ledger_reports_reuse(self):
        proj = make(self.tmp.name, "diverse")
        self.assertEqual(run("verdict", str(proj))[0], 1)
        run("blend", str(proj))
        self.assertNotIn("WARN", run("verdict", str(proj), "--boot", "200")[1])
        run("blend", str(proj), "--method", "weights")           # a second recipe
        self.assertIn("already scored 1 other recipe", run("verdict", str(proj), "--boot", "200")[1])
        self.assertEqual(len(json.loads((proj / "holdout-ledger.json").read_text())), 2)
        self.assertEqual((proj / "RECORD.md").read_text().count("## "), 4)

    def test_multiclass(self):
        rng = np.random.default_rng(3)
        n, proj = 1500, Path(self.tmp.name) / "m"
        y = rng.integers(0, 3, size=2 * n)
        labels = ["cat", "dog", "eel"]
        for m in ("a", "b", "c"):
            logits = rng.normal(size=(2 * n, 3))
            logits[np.arange(2 * n), y] += 1.5
            p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            hdr = ["p_" + l for l in labels]
            write(proj / "oof" / f"{m}.csv", ["id", "fold"] + hdr, [[i, i % 5] + list(p[i]) for i in range(n)])
            write(proj / "holdout" / f"{m}.csv", ["id"] + hdr, [[i] + list(p[i]) for i in range(n, 2 * n)])
        write(proj / "truth.csv", ["id", "y"], [(i, labels[y[i]]) for i in range(n)])
        write(proj / "holdout_truth.csv", ["id", "y"], [(i, labels[y[i]]) for i in range(n, 2 * n)])
        (proj / "ensemble.json").write_text(json.dumps({"task": "multiclass", "min_gain": 0.01}))
        self.assertEqual(run("check", str(proj))[0], 0)
        self.assertEqual(run("members", str(proj))[0], 0)
        run("blend", str(proj))
        self.assertIn("VERDICT: KEEP", run("verdict", str(proj), "--boot", "200")[1])

    def test_lint_flags_the_classic_leak(self):
        f = Path(self.tmp.name) / "stack.py"
        f.write_text("X = scaler.fit_transform(X)\nX_train, X_test = train_test_split(X)\n"
                     "meta_X = np.column_stack([m.predict_proba(X_train)[:, 1] for m in models])\n"
                     "cal = CalibratedClassifierCV(m, cv='prefit')\n")
        code, out = run("lint", str(f))
        self.assertIn("stack.py:1", out)
        self.assertIn("stack.py:3", out)
        self.assertIn("FAIL", out)


if __name__ == "__main__":
    unittest.main()
