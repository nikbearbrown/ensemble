# Rules for ensembles that earn their place

An ensemble works for one reason: if several competent models make *different* mistakes, combining them cancels some of those mistakes. A competent model that makes the same mistakes as the others adds nothing. A different model that is not competent adds noise. An ensemble chosen or weighted with information from the test set only looks good.

So an ensemble has to earn its extra complexity with a measured improvement over the strongest practical single model, on the outcome that matters: accuracy, probability quality, stability, a subgroup, or reliability when the data changes.

Most ensembles that disappoint in use were not badly combined. They were badly evaluated. Parts 3, 6 and 7 are the ones to read twice.

Scope: supervised classification, regression and forecasting, including tree ensembles and neural-network ensembles.

This document merges two rule sets Nik Bear Brown supplied (one of 18 rules, one of 50) and three additions from his notes on other people's ensemble skills. Where the two sets disagreed, the choice made is in "Where the sources disagreed" at the end. The rule numbers here are the ones the skills and the `./ens` tool cite.

---

## Part 1. Decide what the ensemble is for

**Rule 1. State the prediction task exactly.** The target, the unit predicted, the moment of prediction, and the population. "Predict churn" is incomplete. "Whether an active customer cancels within 30 days, from what is known at the start of that period" tells you which features, splits and metrics are defensible.

**Rule 2. Choose the primary metric before comparing models.** It should match the decision: log loss or Brier score for probabilities, precision at a required recall, absolute error for quantities, or a stated cost. Write down the secondary metrics and limits too, so a gain in one number cannot hide damage elsewhere.

**Rule 3. Say what improvement would matter.** Fix the smallest gain worth the extra training, serving, maintenance and debugging, before any ensemble score exists. A tiny average gain that is unstable across splits is not worth it. A modest gain that reliably removes a costly error may be. (`min_gain` in `ensemble.json`.)

**Rule 4. Set a budget.** Latency, memory, training time, cost, and how many components the team can maintain. These are selection criteria from the start, so an ensemble that looks good statistically can be rejected for not fitting its use.

**Rule 5. Decide whether to ensemble at all.** If one model clearly dominates and the others make the same mistakes, use that model.

## Part 2. Baseline and method

**Rule 6. Beat a well-tuned single model first.** The baseline is the best single model, tuned with the same budget, the same information and the same validation protocol as the ensemble, and chosen on development data. If it is not tuned, the ensemble takes credit for what tuning would have given. A random forest or a gradient-boosted model is already an ensemble; it is still the bar.

**Rule 7. Know which error you are attacking.**

| Method | Mainly reduces | Works best with | Does little for |
|---|---|---|---|
| Bagging, random forests | Variance | Unstable, low-bias learners (deep trees, neural nets) | Stable learners (linear models, kNN with large k) |
| Boosting | Bias, some variance | Weak, high-bias learners (shallow trees) | Already-complex learners, which it tends to overfit |
| Voting, averaging, blending, stacking | Errors of mismatched model families | Different families with different inductive biases | Near-identical models |
| Deep ensembles (seeds) | Variance and over-confidence | Neural nets with non-convex losses | Convex or deterministic models, which land on the same solution |

If the model underfits, bagging it will not help. If it overfits badly, boosting it will make things worse. Weak trees are the right members for boosting; a hand-built voting ensemble needs a different reason for each member.

## Part 3. Protect the data boundaries

**Rule 8. Split first.** Set the training and evaluation partitions before fitting anything whose parameters come from data. Take the final holdout off before anything else.

**Rule 9. Keep time order when predicting the future.** Use forward-chaining (rolling-origin) splits. Account for when features and labels actually become available. With overlapping label windows, purge and leave a gap (López de Prado). The same restriction applies to member tuning, stacking, calibration and threshold selection.

**Rule 10. Split at the unit that must generalise.** New patients, users, sites or documents: hold out whole groups. Stratifying preserves class proportions; it does nothing about dependence. Keep duplicates, near-duplicates, crops of one image, excerpts of one document and augmented copies on one side.

**Rule 11. Fit every member's whole preprocessing pipeline inside each fold.** Scalers, imputers, encoders (target encoders above all), feature selection, dimensionality reduction. Package each member with its own transformations so training, validation and serving run the same procedure.

**Rule 12. Oversample only inside training partitions.** SMOTE and its relatives run after the split, on the training fold. Evaluation rows keep the class balance of real use, or a stated weighting.

**Rule 13. Choosing an ensemble is model selection, and selection costs a holdout.** Which members, which weights, which combiner, which calibration, which threshold, when to stop: all are fitted choices. Keep the final evaluation data outside every one of them. Trying alternatives against the same evaluation result overfits the selection even when each model was trained correctly (Cawley and Talbot, 2010). Touch the holdout once. If a test result informed a revision, say so and get fresh evidence for the revised system.

**Rule 14. Use nested evaluation when data is short, and choose folds from the evidence you have.** An outer loop that reruns the whole procedure, tuning included, estimates the procedure honestly when one holdout would be too small to trust. Five or ten folds are conventions. Count the independent units and the rare outcomes in each fold before settling.

## Part 4. Members

**Rule 15. Diversity has to come from somewhere real.** Roughly strongest to weakest:

- different model families (boosted trees, linear models, neural nets, kNN)
- different feature representations or preprocessing
- different training data (bootstrap samples, subsamples, time windows)
- different feature subsets
- different hyperparameters
- different random seeds (effective for neural nets, close to useless for convex or deterministic models)

Different algorithm names alone establish nothing.

**Rule 16. Measure diversity on held-out errors, and do not worship the number.** Look at out-of-fold residuals (regression) or jointly wrong predictions, per-class errors and probability errors (classification). Raw prediction correlation misleads, because competent models should agree on most of the signal. For an equal-weight average of M regressors (Ueda and Nakano, 1996):

```
ensemble MSE = (average bias)^2 + (1/M)(average variance) + (1 - 1/M)(average covariance)
```

More members shrink the variance term. They do not shrink the covariance term. If M members have the same error variance and pairwise error correlation rho, the average keeps `rho + (1 - rho)/M` of one member's variance: ten models at rho = 0.8 keep 82%. This is an illustration for squared error under simplifying assumptions, not a formula for accuracy.

Krogh and Vedelsby's decomposition says ensemble error equals average member error minus average disagreement. It promises the ensemble beats the *average* member, not the *best* one. The same is true of Jensen's inequality for convex losses.

Kuncheva and Whitaker (2003) found no diversity measure that reliably predicts ensemble accuracy. Use correlation to find redundant members, not as a quantity to maximise. Maximising diversity rewards models that are different because they are wrong.

**Rule 17. Every member has to show what it contributes.** Evaluate each candidate as a complete system: score, calibration, error pattern, cost. A member barely better than chance adds mostly noise unless the method is built for weak learners, as boosting is. A weaker model may still add something the others lack; require evidence of it. Look at *where* members disagree: hard cases, rare classes, a subgroup, or noise.

**Rule 18. Remove members and watch what happens; watch the size curve.** Drop each member and measure the effect on the metric, the slices, calibration and cost, refitting the combiner on development data. Plot performance against the number of members. The curve usually flattens fast. Stop adding when the gain stops mattering: every member kept costs inference time and maintenance for as long as the system lives.

## Part 5. Combining

**Rule 19. Start with a simple average.** Equal weights are hard to beat. Forecasting research calls this the forecast combination puzzle: simple averages often outperform "optimal" weights because estimating the weights adds its own error. Learned weights or a stacker must beat the simple average out of sample before they are adopted.

**Rule 20. Align predictions before combining them.** Row ids, class-label order, target definition, units, forecast horizon, output shape. Store the ids with the predictions. Two arrays can have the same shape and refer to different rows or opposite classes; the result is a score near chance and no error message.

**Rule 21. Combine on one scale, and respect the output's structure.** Averaging probabilities (soft voting) usually beats majority voting, but only if the probabilities are comparable. SVM margins, boosting scores and softmax outputs are not on one scale: calibrate each member first, or combine ranks. Averaging probabilities, averaging logits and voting are three different systems; choose one and use the same one in serving. Regression: the mean by default, the median when a member sometimes produces wild values; undo target transformations first. Never average class identifiers as numbers. Averaging interval endpoints does not establish coverage.

**Rule 22. Constrain the combiner.** For a mixture of probabilities, non-negative weights that sum to one. For stacking, start with a simple, regularised combiner: Breiman's stacked regressions found non-negative linear weights worked well. A flexible combiner (a deep tree, a boosted model) learns the quirks of the out-of-fold table and needs enough data to justify it. Adding original features to the stack adds capacity and is a separate choice to evaluate.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
meta_reg = LinearRegression(positive=True)   # non-negative weights
meta_clf = LogisticRegression(C=0.1)         # strong regularisation
```

## Part 6. Stacking and learned weights

**Rule 23. Train the combiner only on out-of-fold predictions.** The cardinal rule. If the combiner sees predictions on rows the base models trained on, it learns to trust whichever model memorised the training data best.

```python
# WRONG: in-sample predictions feed the combiner
for m in base_models:
    m.fit(X_train, y_train)
meta_X = np.column_stack([m.predict_proba(X_train)[:, 1] for m in base_models])
meta.fit(meta_X, y_train)

# RIGHT: out-of-fold predictions, the same folds for every base model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_X = np.column_stack([
    cross_val_predict(m, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    for m in base_models
])
meta.fit(meta_X, y_train)
for m in base_models:
    m.fit(X_train, y_train)   # refit on all training data for inference
```

scikit-learn's `StackingClassifier` and `StackingRegressor` do this internally through `cv`, and warn about `cv="prefit"`. Hand-rolled stacking is where the leak usually happens.

**Rule 24. Tuning and early stopping are inside the boundary too.** For out-of-fold predictions to be clean, the held-out rows must also be excluded from hyperparameter selection, supervised feature selection, early stopping and member calibration. Hyperparameters chosen on the same folds that produce the predictions leak a little. With small data, nest.

**Rule 25. One set of folds, with the data's structure kept.** Every base model uses the same fold assignments. Group and time restrictions hold at the stacking level as well. For forecasts, each row's prediction comes from a model trained only on what was available at that forecast origin, and the combiner trains only on rows that have such a prediction. `cross_val_predict` needs every row held out exactly once, so rolling splits and repeated cross-validation need their predictions assembled by hand.

**Rule 26. The out-of-fold table is the combiner's training data.** A score from fitting weights or a stacker on that table and scoring the same table is a training score. Cross-validating the combiner over the saved table is better and still a diagnostic: the base models behind each row were shaped by the other rows' labels, and you are choosing a recipe with the result. The number to report comes from data outside the whole construction: an untouched holdout, or an outer loop that rebuilds everything, base-model folds included, inside each outer training split. When reporting cross-validated metrics, say whether they were computed per fold or pooled.

**Rule 27. Decide how the final models are fitted, and evaluate that.** Usual: refit each base pipeline on all development data and keep the combiner trained on out-of-fold predictions. Alternative: keep the fold-trained models and average them. Blending (one held-out slice instead of folds) is simpler and harder to get wrong but wastes data; reasonable with a lot of it. Whatever is deployed is what must have been evaluated.

## Part 7. Evaluation

**Rule 28. Report the gain with its uncertainty, on the same cases for every candidate.** Compare paired, against the baseline chosen in development, not whichever baseline flatters the result. A 0.3% improvement averaged over five folds may be noise. Report the spread across folds. Cross-validation folds share training data, so a plain paired t-test understates the variance; use a test that accounts for it, such as Nadeau and Bengio's corrected resampled t-test, and remember no unbiased estimator of k-fold variance exists (Bengio and Grandvalet, 2004). Resample whole units when rows are clustered. Report cost beside accuracy.

**Rule 29. Check calibration on the combined output.** Combining changes calibration. Boosted models tend to be poorly calibrated to begin with (Niculescu-Mizil and Caruana, 2005). Averaging calibrated members does not give a calibrated average; a linear pool of calibrated forecasts tends to be underconfident (Gneiting and Ranjan, 2013). If anyone uses the probabilities, look at reliability by bin with counts, and at proper scores (log loss, Brier). A better Brier score alone does not show better calibration.

**Rule 30. A calibrator or a threshold needs its own clean evidence.** A calibrator fitted after stacking must train on predictions from an ensemble that did not train on those labels: a separate slice, or cross-fitting the whole stack. Cross-fitting only the base models is not enough if the combiner then trains on all the labels. Choose decision thresholds on development data from the costs involved, and freeze them before the final evaluation.

**Rule 31. Disagreement is a signal, not a calibrated probability.** Spread across members is a useful trigger for human review. It is not a confidence interval, a posterior, or a probability of error. Members trained on the same data can agree confidently on inputs far from anything they have seen, and spread in predicted means leaves out observation noise. Validate uncertainty on shifted data before relying on it, and evaluate any abstain-or-escalate policy as a whole: how many cases are sent on, how many errors remain, for whom.

**Rule 32. Look at slices and at consequential errors.** Classes, subgroups, sites, input quality, with counts and uncertainty so small slices are not over-read. Read the confident mistakes and the cases every member gets wrong.

**Rule 33. Test realistic shift.** Later periods, new sites, changed missingness, corrupted inputs, the failure of a feature or source all members share. Calibration fixed on the original distribution does not hold after it moves (Ovadia et al., 2019).

## Part 8. Use and honesty

**Rule 34. Pay the costs on purpose.** An ensemble multiplies latency, memory, monitoring and retraining, and makes predictions harder to explain. Measure the whole inference path: preprocessing, every model, aggregation, calibration, decision logic. Running members in parallel changes scheduling; it does not make the cost go away. If the gain is worth having but the ensemble is too expensive to serve, distil it into one model (Buciluǎ et al., 2006; Hinton et al., 2015) and evaluate that model as a new system. Pruning, replacing, compressing or distilling all create a changed system that needs its evaluation again.

**Rule 35. An ensemble cannot fix what all its members share.** If every member learned from the same biased, mislabelled or unrepresentative data, they share the blind spot and averaging keeps it. Ensembles reduce disagreement between models. They do not correct systematic error in what all of them learned from.

**Rule 36. Keep the system reproducible, monitored, and defined when a part fails.** Version together: the data snapshot, the split definitions, the feature code, each member and its preprocessing, the out-of-fold predictions that trained the combiner, the weights, the calibrator, the thresholds, the seeds and the environment. Train-serve skew is worse in ensembles: a mismatch in one member corrupts the output and hides inside the average. Say what happens when a member times out or loses a feature; silently dropping it and renormalising is a different model. Monitor the members as well as the ensemble, and the upstream sources they share. Keep a record of what was tried, including what failed.

---

## Notes by method

**Bagging and random forests.** Use deep, unpruned trees: bagging needs variance to average away. `max_features` sets the trade: Breiman's bound on forest error depends on the strength of the trees and the correlation between them, and fewer features per split lowers both. Adding trees does not overfit; it stops helping. The out-of-bag error shows where.

**Boosting.** Tune learning rate and number of rounds together; a lower rate needs more rounds and usually generalises better. Always stop early on validation data: unlike forests, boosting does overfit with too many rounds. Watch label noise: AdaBoost's exponential loss keeps raising the weight on mislabelled points, and Long and Servedio (2010) showed random label noise can defeat boosters built on a convex potential. On noisy data use robust losses, row subsampling and strong regularisation.

**Stacking and blending.** Rules 22 to 27.

**Forecasts.** Start with the equal-weight average of three to five forecasts made by different methods. Learned weights need a long backtest to pay for themselves. Every weight, stacker and calibrator is fitted on backtest forecasts made strictly before the period it is used on.

**Deep ensembles.** Independent random initialisations and data shuffling are usually enough. Lakshminarayanan et al. (2017) found bootstrapping the data made deep ensembles worse, because each member saw less of it. Cheaper approximations (snapshot ensembles, stochastic weight averaging, MC dropout) give less diverse members; Ovadia et al. (2019) found full deep ensembles held up best under shift.

## Checklist

- [ ] The target, prediction time, population and allowed inputs are written down.
- [ ] The primary metric and the smallest gain worth having were fixed in advance.
- [ ] A tuned single-model baseline exists, scored with the same protocol.
- [ ] The method matches the error (variance: bagging; bias: boosting; mismatched families: stacking).
- [ ] Splits reflect entities, time order and real use.
- [ ] Preprocessing, sampling, tuning and early stopping sit inside those splits.
- [ ] Diversity has a real source; held-out error correlations were looked at; each member survives removal.
- [ ] Predictions are aligned by id and on one scale.
- [ ] A simple average was tried before learned weights.
- [ ] The combiner trained only on out-of-fold predictions and is constrained.
- [ ] The stack was scored outside the table that trained its combiner.
- [ ] Member selection and weighting never touched the final holdout.
- [ ] The gain is reported with fold spread and an interval, and slices were examined.
- [ ] Calibration and thresholds were fitted and checked on clean data, if probabilities are used.
- [ ] Shift, shared failures and a member going down were examined.
- [ ] Serving cost, explainability and maintenance were weighed against the gain.
- [ ] The whole system can be rebuilt, monitored, updated and rolled back.

## Where the sources disagreed

- **"Every member must be competent" against "a weaker member may still contribute".** Kept both halves as rule 17: a member needs evidence of contribution, and standalone score is one piece of it.
- **"Drop one of any pair with residual correlation above about 0.95" against "avoid fixed correlation thresholds".** The tool flags pairs above 0.95 as places to look. Removal is decided by the ablation (rule 18).
- **"Averaging calibrated members makes the ensemble underconfident" against "averaging does not in general preserve calibration".** Rule 29 states the general claim and gives underconfidence as the usual direction for a linear pool.

Added from Nik's notes on other skills (see `docs/prior-art.md`): align by row id (rule 20), a blend scored on the rows that fitted its weights is a diagnostic (rule 26), and rolling backtests for forecasts (rules 9 and 25).

## Sources

The two supplied rule sets cited these. They were not re-read for this repo, apart from the scikit-learn pages.

- Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of k-fold cross-validation. *JMLR* 5. https://www.jmlr.org/papers/v5/grandvalet04a.html
- Breiman, L. (1996). Bagging predictors. *Machine Learning*.
- Breiman, L. (1996). Stacked regressions. *Machine Learning*.
- Breiman, L. (2001). Random forests. *Machine Learning*. https://www.stat.berkeley.edu/~breiman/random-forests.pdf
- Buciluǎ, C., Caruana, R., & Niculescu-Mizil, A. (2006). Model compression. *KDD*.
- Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. *ICML*.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 11. https://www.jmlr.org/papers/v11/cawley10a.html
- Dietterich, T. G. (2000). Ensemble methods in machine learning. *Multiple Classifier Systems*.
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *JCSS*.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*.
- Gneiting, T., & Ranjan, R. (2013). Combining predictive distributions. *Electronic Journal of Statistics*. https://arxiv.org/abs/1106.1638
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *NeurIPS*.
- Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles. *Machine Learning*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Long, P. M., & Servedio, R. A. (2010). Random classification noise defeats all convex potential boosters. *Machine Learning*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? *NeurIPS*.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*.
- Ueda, N., & Nakano, R. (1996). Generalization error of ensemble estimators. *ICNN*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*.
- Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*.
- scikit-learn: [ensembles](https://scikit-learn.org/stable/modules/ensemble.html), [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html), [decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html). imbalanced-learn: [common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).
