# Domain 4 — Heart Disease Prediction (Tabular ML / Classification)

> 🔄 SOTA ล่าสุด (กลางปี 2026) + library เจ๋ง ๆ ดู `../ADVANCED_SOTA.md` และ `../LIBRARIES.md` — ไฟล์นี้คือความรู้พื้นฐาน/กับดักที่ยังใช้ได้

Competition: `super-ai-engineer-ss-6-heart-disease-prediction`

Source note: Kaggle page is invitation-only / login-walled. Reconstructed from the UCI Heart Disease schema (canonical source), SS5 precedent, and sibling Kaggle heart comps. Confirm the metric on the Evaluation tab before optimizing.

## A. TASK SUMMARY
- Task: binary classification — predict presence (`1`) vs absence (`0`) of heart disease. (UCI raw `num` 0-4; standard binarize `>0 -> 1`.)
- Features (~13 UCI columns, mixed types):
  - Numerical: `age`, `trestbps` (resting BP), `chol`, `thalach` (max HR), `oldpeak` (ST depression).
  - Categorical/ordinal: `sex`, `cp` (chest pain, 4), `fbs`, `restecg` (3), `exang`, `slope` (3), `ca` (0-3 vessels), `thal` (normal/fixed/reversible).
- Size: small — Cleveland ~303 rows, full UCI ~920. This is TabPFN territory (≤10k rows). Missing values common in `ca`, `thal` (raw `?`).
- Target: `target`/`num` (0/1 after binarize).
- Metric (verify): one of `Accuracy`, `AUC-ROC`, `F1`. `P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2PR/(P+R)`, `AUC=P(score(pos)>score(neg))`.
- Submission: `id,target` CSV. AUC → submit probabilities; Accuracy/F1 → submit hard 0/1.

## B. DOMAIN CATEGORIZATION
- Supervised → tabular → binary classification.
- Medical/clinical ML: small-n, mixed cat+continuous, informative missing values, interpretability matters, imbalance common, false negatives costly → recall/F1 often weighted over accuracy.
- Best family for n<10k: gradient-boosted trees (LightGBM/XGBoost/CatBoost) + TabPFN v2. Deep tabular nets usually don't beat GBDTs here.

## C. PREDICTED EXAM VARIATIONS
1. Multiclass severity (`num` 0-4). `objective=multiclass`, metric macro-F1 / multiclass log-loss / QWK. StratifiedKFold on 5 classes, argmax (no threshold). CatBoost clean.
2. Regression risk score. `objective=regression`/RMSE, metric RMSE/R². Drop threshold tuning; clip range.
3. Imbalanced / fraud-style. Metric AUC-PR / F1 / MCC. `scale_pos_weight = n_neg/n_pos`, threshold on PR curve, SMOTE only inside folds.
4. Survival / time-to-event. Cox PH / `lifelines` / `scikit-survival` `cox`/`aft`. Metric C-index. Classification doesn't apply.
5. Calibration / cost-sensitive. Metric log-loss / Brier. `CalibratedClassifierCV` (isotonic/sigmoid).
6. Large synthetic tabular (100k+). TabPFN slow → LightGBM/CatBoost heavier tuning. Usually AUC.

Decision rule: probability metric (AUC/log-loss/Brier) → submit `predict_proba`. Hard-label metric (Acc/F1/MCC) → thresholded labels, tune threshold on OOF.

## E. SOTA TABLE (small medical tabular, n<10k)
| Rank | Model | Library | When to use | Key knobs |
|---|---|---|---|---|
| 1 | TabPFN v2 / v2.5 | `tabpfn` | n≤10k, ≤500 feats | ~zero tuning, auto NaN/cat/scale, `AutoTabPFNClassifier`, `device` |
| 2 | CatBoost | `catboost` | many categoricals, missing | `depth`(4-8), `learning_rate`, `l2_leaf_reg`, early stop |
| 3 | LightGBM | `lightgbm` | speed, baseline | `num_leaves`, `min_child_samples`, `scale_pos_weight` |
| 4 | XGBoost | `xgboost` | strong baseline, ensemble diversity | `max_depth`(3-6), `eta`, `reg_lambda` |
| 5 | Logistic Regression | `sklearn` | linear sanity baseline | `C`, `penalty`, `class_weight` |
| 6 | AutoGluon Tabular | `autogluon` | EASIEST — auto-ensembles all above | `presets='best_quality'`, `time_limit` |
| 7 | FT-Transformer / SAINT | `rtdl` | larger tabular, research | depth, lr, dropout |
| 8 | RandomForest/ExtraTrees | `sklearn` | quick robust (won SS5) | `n_estimators`, `class_weight` |

Best single move under time pressure: AutoGluon `best_quality` (one call), or TabPFN for n<10k, then blend with CatBoost.

## F. PRACTICAL GOTCHAS
- Leakage: fit Imputer/Scaler/Encoder/SMOTE inside each fold only. Drop `id`. Watch target-derived columns.
- SMOTE only on training fold, never before split. On tiny data `scale_pos_weight`/`class_weight` often beats SMOTE.
- Always `StratifiedKFold(shuffle=True, random_state=...)` on 300 rows.
- Don't scale one-hot meaningfully; don't label-encode nominal as ordinal for linear/NN. Let CatBoost/TabPFN ingest cats natively.
- Missing values informative (`ca`,`thal`): GBDTs/TabPFN handle NaN natively; consider a missing-indicator column.
- Pick final submission by CV OOF, not noisy public LB.
- Threshold/metric mismatch: tune threshold on OOF for F1/Acc; submit raw probs for AUC/log-loss; calibrate for Brier.

## G. QUICK-WIN BASELINE (<30 min)
LightGBM + StratifiedKFold(5) + `scale_pos_weight` + early stopping, OOF + averaged test preds. LightGBM eats NaN + raw ints (skip ColumnTransformer). F1/Acc → tune threshold on OOF; AUC → submit probs. Then add CatBoost + TabPFN and average (+0.5-2 pts). Or just run AutoGluon `best_quality` for a hands-off strong baseline.

## H. KEYWORDS TO STUDY
```
UCI Heart Disease, Cleveland Hungary Switzerland VA, binary vs multiclass num target,
StratifiedKFold, out-of-fold OOF, cross-validation leakage,
LightGBM XGBoost CatBoost, gradient boosting hyperparameters, early stopping,
scale_pos_weight, class_weight, SMOTE, imbalanced-learn, class imbalance,
sklearn ColumnTransformer Pipeline SimpleImputer OneHotEncoder StandardScaler,
missing value imputation, missing indicator, native NaN handling in GBDT,
TabPFN v2 TabPFN-2.5 AutoTabPFNClassifier, tabular foundation model, in-context learning,
AutoGluon TabularPredictor best_quality, FT-Transformer SAINT TabNet,
threshold tuning, F1 optimization, precision recall tradeoff, AUC-ROC AUC-PR MCC,
probability calibration CalibratedClassifierCV isotonic Brier log-loss,
model ensembling stacking blending weighted average, trust your CV,
SHAP feature importance permutation importance, survival analysis Cox concordance index,
quadratic weighted kappa, macro-F1, multiclass log-loss
```
