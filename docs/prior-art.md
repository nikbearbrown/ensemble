# Prior art

Other people's skills for ensembles, found through Nik Bear Brown's index of public agent skills, and what this repo took from each. Ideas were taken; no text or code was copied. Nothing here was installed or run.

## Read

**[ensemble-submit](https://github.com/Alexdruso/data-science-stuff/blob/main/.claude/skills/ensemble-submit/SKILL.md)** and its [stacking.py](https://github.com/Alexdruso/data-science-stuff/blob/main/data_science_stuff/kaggle/stacking.py) (Alexdruso/data-science-stuff). A Kaggle workflow: load saved out-of-fold and test predictions, fit blend weights with Nelder-Mead, stack with logistic regression when the blend plateaus, greedy selection inside an outer cross-validation.

- Taken: predictions are keyed by row id and re-ordered on load, with an error on any missing, extra or duplicate id. Their skill records that a row-order mismatch once shipped a score near chance. That is rule 20 and most of `./ens check`.
- Taken: check that members share folds before combining.
- Taken: error overlap between members says more about blend lift than correlation of raw probabilities; when a stack is flat, the missing piece is a different kind of member, not a better combiner.
- Not taken: the blend is scored on the same out-of-fold rows that fitted its weights, and the verification step is "blended score at least the best single score" on those rows. Here that is a training score (rule 26). `./ens blend` fits each fold's recipe on the other folds, calls even that a diagnostic, and leaves the claim to `./ens verdict`.
- Not taken: stacking over the saved prediction table with fresh random folds is described there as honest. It is honest about the combiner's fit. The base models behind each row still learned from the other rows.
- Not taken: tuning to a public leaderboard, per-class threshold weights fitted on the blended predictions, submission steps.

**[algo-forecast-ensemble](https://github.com/asgard-ai-platform/skills/blob/main/algo-forecast-ensemble/SKILL.md)** (asgard-ai-platform/skills). Combining forecasts: equal weights, inverse-error weights, stacking.

- Taken: start with the equal-weight average and move to learned weights only with plenty of validation data (rule 19); do not ensemble when one model clearly dominates (rule 5).
- Not taken: the gate "ensemble error no worse than the best single model" with no holdout named and no interval. Here the bar is `min_gain`, on rolling backtests for development and the most recent period as holdout.

**scikit-learn documentation**: [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) (out-of-fold predictions through `cv`, the `prefit` warning, base models refitted on all data) and [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Listed, not read

- [sklearn-ensemble](https://github.com/11vated/Nexus/blob/main/agent-system/.claude/skills/sklearn-ensemble/SKILL.md) (11vated/Nexus): reported to be a stub.
- [scikit-learn skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scikit-learn/SKILL.md) (K-Dense-AI/scientific-agent-skills): a general scikit-learn reference.
- Also in the index under ensemble names, unread: `htkr/opencode-agent-profiles` (kaggle-ensemble-planner), `sadjad6/open-autonomous-research-lab` (ensemble_builder), `snoodleboot-io/prompticorn` (ensemble-methods), `LgrappaG/Workflows-Agents` (ensemble-methods), `deesatzed/UROK` (ensemble_review).
