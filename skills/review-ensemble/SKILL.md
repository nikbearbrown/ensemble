---
name: review-ensemble
description: Reviews ensemble, blending or stacking code, a notebook, or a claimed ensemble result against the rules in docs/rules.md, with leakage first. Part of the ensemble skill; read the root SKILL.md first.
---

# Review an ensemble

A review answers one question: is the claimed gain over a single model real, and is it worth its cost? Read the code before the results. Rule numbers refer to `docs/rules.md`.

## Procedure

1. Run `./ens lint <path>` and read every line it points to. It matches text patterns. A hit is a place to look, and a clean run is not a pass.
2. Trace the data. For every number in the results, write down which rows trained the base models, which rows trained the combiner, which rows chose anything (members, weights, hyperparameters, thresholds, stopping rounds), and which rows produced the number. If two of those sets overlap, name the overlap.
3. If the project saves predictions, put them in the file layout (`docs/file-format.md`) and run `./ens check`, `members` and `blend`. Compare what you measure with what was claimed.
4. Go through the list below. Report findings in its order: a leak makes everything after it unreadable.

## What to look for, most serious first

**Leaks that inflate the score**

- The combiner, the weights or a calibrator were fitted on predictions the base models made on their own training rows (rule 23). In scikit-learn, `cv="prefit"` with the same rows is this.
- Scaling, imputing, encoding, feature selection or oversampling were fitted before the split, or outside the fold (rules 11 and 12).
- Rows from one entity, or near-duplicates, sit on both sides of a split (rule 10).
- Random folds on a task that predicts the future, or a stacker with random folds on top of time-ordered base models (rule 9).
- Members did not share folds, so one member's training rows are another's validation rows.

**Scores that are not what they are called**

- The blend is scored on the rows that fitted its weights. That is a training score for the combiner.
- A stacker is cross-validated over a saved table of out-of-fold predictions and the result is called held-out. The base models behind each row have seen the other rows' labels; it is a diagnostic (rule 26).
- Members, weights or thresholds were chosen by looking at the test set, then the test score is reported (rule 13). Count how many times the test set is scored.
- Base models were tuned or early-stopped on the folds that produced their out-of-fold predictions (rule 24).

**A weak comparison**

- No single-model baseline, or one that was not tuned with the same effort (rule 6).
- Candidates scored on different rows, or with different metrics.
- A gain reported as one number: no fold spread, no interval, or a plain paired t-test across folds (rule 28).
- A gain smaller than the fold-to-fold spread, with no statement of what gain would matter.

**A combination that cannot work as written**

- Predictions aligned by position instead of id; class columns in different orders (rule 20).
- Margins, raw scores and probabilities averaged together; class labels averaged as numbers (rule 21).
- A flexible combiner (boosted trees, a deep network) on a small table of predictions, never compared with a plain average (rules 19 and 22).
- Members that are the same model with a different seed where the seed changes nothing; no ablation (rules 15 to 18).

**What is missing for use**

- No calibration check on the combined output when probabilities are used (rule 29).
- Member disagreement presented as a confidence interval or a probability of error (rule 31).
- No slices, no shift test, no cost, no plan for a member failing (rules 32 to 36).

## The report

Start with one sentence: whether the claimed gain is supported, not supported, or cannot be judged from what exists. Then the findings, each with the file and line, the rule, what it does to the number, and the fix. Separate what you ran from what you read. End with the smallest set of steps that would settle the question, which is usually: rebuild the predictions inside proper folds, lock one recipe, score it once on rows nothing has touched.

Do not soften a leak into a "consideration". Do not call a result wrong when it is only unproven: say which.
