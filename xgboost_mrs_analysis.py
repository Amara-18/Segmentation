#!/usr/bin/env python3
"""
XGBoost-based mRS outcome analysis.

This script builds four predefined models (M1-M4) for mRS 0-2 vs 3-6,
performs MRMR + LASSO feature selection for radiomics features, and
exports the tables and figures needed for manuscript reporting.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, lasso_path
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*old version of glibc.*")


RANDOM_STATE = 2
MRMR_TOP_K = 50
XGB_RANDOM_SEARCH_ITER = 18
UNIVARIATE_P_THRESHOLD = 0.05
MULTIVARIATE_P_THRESHOLD = 0.05
BASE_MODEL_NAMES = ["M1", "M2", "M3", "M4"]


def is_valid_fusion_combo(combo: tuple[str, ...] | list[str]) -> bool:
    return "M4" not in combo or {"M2", "M3"}.issubset(combo)


FUSION_MODEL_NAMES = [
    f"Fusion_stack_{'_'.join(combo)}"
    for size in range(2, len(BASE_MODEL_NAMES) + 1)
    for combo in combinations(BASE_MODEL_NAMES, size)
    if is_valid_fusion_combo(combo)
]
FINAL_FUSION_MODEL_NAME = "Fusion_stack_M1_M2_M3_M4"

ROI_PREFIX_VOLUME_COLUMNS = {
    "pre_IVH_": ["pre_IVH_volume", "pre_IVH_volume_cm3"],
    "post_IVH_": ["post_IVH_volume", "post_IVH_volume_cm3"],
    "pre_SAH_": ["pre_SAH_volume", "pre_SAH_volume_cm3"],
    "post_SAH_": ["post_SAH_volume", "post_SAH_volume_cm3"],
    "pre_Ventricle_": ["pre_Ventricle_volume", "pre_Ventricle_volume_cm3"],
    "post_Ventricle_": ["post_Ventricle_volume", "post_Ventricle_volume_cm3"],
    "pre_CompleteVent_": ["pre_CompleteVent_volume"],
    "post_CompleteVent_": ["post_CompleteVent_volume"],
    "pre_total_ventricular_": ["pre_total_ventricular_volume_cm3", "pre_total_ventricular_volume"],
    "post_total_ventricular_": ["post_total_ventricular_volume_cm3", "post_total_ventricular_volume"],
}

DELTA_ROI_VOLUME_COLUMNS = {
    "delta_IVH_": (["pre_IVH_volume", "pre_IVH_volume_cm3"], ["post_IVH_volume", "post_IVH_volume_cm3"]),
    "delta_SAH_": (["pre_SAH_volume", "pre_SAH_volume_cm3"], ["post_SAH_volume", "post_SAH_volume_cm3"]),
    "delta_Ventricle_": (
        ["pre_Ventricle_volume", "pre_Ventricle_volume_cm3"],
        ["post_Ventricle_volume", "post_Ventricle_volume_cm3"],
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    role: str


@dataclass
class FeatureSelectionResult:
    selected_radiomics: list[str]
    mrmr_features: list[str]
    lasso_nonzero: list[str]
    lasso_coef: pd.DataFrame
    lasso_cv: pd.DataFrame
    rfe_curve: pd.DataFrame
    rfe_ranking: pd.DataFrame


@dataclass
class ClinicalSelectionResult:
    selected_clinical: list[str]
    univariate_table: pd.DataFrame
    multivariate_table: pd.DataFrame
    rfe_curve: pd.DataFrame
    rfe_ranking: pd.DataFrame


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def apply_structural_roi_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Set radiomics from absent ROIs to zero before any statistical imputation."""
    df = df.copy()
    for prefix, volume_candidates in ROI_PREFIX_VOLUME_COLUMNS.items():
        volume_col = first_existing_column(df, volume_candidates)
        if volume_col is None:
            continue
        roi_cols = [c for c in df.columns if c.startswith(prefix)]
        if not roi_cols:
            continue
        volume = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        absent = volume <= 0
        if absent.any():
            df.loc[absent, roi_cols] = df.loc[absent, roi_cols].apply(pd.to_numeric, errors="coerce")
            df.loc[absent, roi_cols] = 0.0

    for prefix, (pre_candidates, post_candidates) in DELTA_ROI_VOLUME_COLUMNS.items():
        pre_col = first_existing_column(df, pre_candidates)
        post_col = first_existing_column(df, post_candidates)
        if pre_col is None or post_col is None:
            continue
        delta_cols = [c for c in df.columns if c.startswith(prefix)]
        if not delta_cols:
            continue
        pre_volume = pd.to_numeric(df[pre_col], errors="coerce").fillna(0)
        post_volume = pd.to_numeric(df[post_col], errors="coerce").fillna(0)
        both_absent = (pre_volume <= 0) & (post_volume <= 0)
        if both_absent.any():
            df.loc[both_absent, delta_cols] = df.loc[both_absent, delta_cols].apply(pd.to_numeric, errors="coerce")
            df.loc[both_absent, delta_cols] = 0.0
    return df


def read_feature_table(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path, encoding="utf-8-sig")
    df = apply_structural_roi_zero_fill(df)
    df["dataset"] = spec.name
    df["dataset_role"] = spec.role
    if "mRS" not in df.columns:
        raise ValueError(f"{spec.path} is missing the mRS column")
    mrs = pd.to_numeric(df["mRS"], errors="coerce")
    df = df.loc[mrs.notna()].copy()
    df["mRS"] = mrs.loc[df.index].astype(int)
    df["mRS_binary"] = (df["mRS"] >= 3).astype(int)
    return df


def load_datasets(specs: list[DatasetSpec]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = [read_feature_table(spec) for spec in specs]
    by_name = {spec.name: frame for spec, frame in zip(specs, frames)}
    train_df = pd.concat([by_name[s.name] for s in specs if s.role == "train"], ignore_index=True)
    return train_df, by_name


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"patient_id", "timestamp", "mRS", "mRS_binary", "dataset", "dataset_role", "subfolder"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() > 0:
            cols.append(col)
    return cols


def classify_feature_groups(columns: list[str], clinical_cols: list[str]) -> dict[str, list[str]]:
    clinical = [c for c in clinical_cols if c in columns]
    pre = [c for c in columns if c.startswith("pre_")]
    post = [c for c in columns if c.startswith("post_")]
    delta_keywords = (
        "delta_",
        "change",
        "clearance",
        "reduction",
        "evolution",
        "recovery",
        "relief",
        "simplification",
        "expansion",
        "efficacy",
    )
    delta = [c for c in columns if c.startswith("delta_") or any(k in c for k in delta_keywords[1:])]
    return {
        "clinical": clinical,
        "pre": pre,
        "post": post,
        "delta": delta,
    }


def prepare_numeric_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def remove_low_information_features(x: pd.DataFrame, max_missing_rate: float = 0.30) -> pd.DataFrame:
    missing_rate = x.isna().mean()
    keep = missing_rate[missing_rate <= max_missing_rate].index.tolist()
    x = x.loc[:, keep]
    nunique = x.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    return x.loc[:, keep]


def impute_and_scale_fit(x: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xi = imputer.fit_transform(x)
    xs = scaler.fit_transform(xi)
    return xs, imputer, scaler


def impute_and_scale_transform(x: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(x))


def mrmr_select(x: pd.DataFrame, y: np.ndarray, top_k: int, random_state: int = RANDOM_STATE) -> list[str]:
    from sklearn.feature_selection import mutual_info_classif

    if x.empty:
        return []
    xs, _, _ = impute_and_scale_fit(x)
    names = np.asarray(x.columns)
    relevance = mutual_info_classif(xs, y, random_state=random_state)
    relevance = np.nan_to_num(relevance, nan=0.0)
    corr = np.corrcoef(xs, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    np.fill_diagonal(corr, 0.0)

    selected_idx: list[int] = []
    remaining = set(range(len(names)))
    top_k = min(top_k, len(names))
    while remaining and len(selected_idx) < top_k:
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            redundancy = float(np.mean(corr[idx, selected_idx])) if selected_idx else 0.0
            score = float(relevance[idx]) - redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected_idx.append(int(best_idx))
        remaining.remove(int(best_idx))
    return names[selected_idx].tolist()


def rfe_one_se_select(
    x: pd.DataFrame,
    y: np.ndarray,
    candidate_features: list[str],
    cv_splits: int = 5,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    candidate_features = [f for f in candidate_features if f in x.columns]
    if not candidate_features:
        return [], pd.DataFrame(columns=["feature", "ranking"]), pd.DataFrame(columns=["n_features", "mean_auc", "std_auc", "se_auc", "selected"])

    x_candidate = x.loc[:, candidate_features]
    xs, _, _ = impute_and_scale_fit(x_candidate)
    estimator = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

    if len(candidate_features) == 1:
        scores = cross_val_score(estimator, xs, y, cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE), scoring="roc_auc")
        curve = pd.DataFrame(
            {
                "n_features": [1],
                "mean_auc": [float(np.mean(scores))],
                "std_auc": [float(np.std(scores))],
                "se_auc": [float(np.std(scores) / np.sqrt(cv_splits))],
                "selected": [True],
            }
        )
        ranking = pd.DataFrame({"feature": candidate_features, "ranking": [1]})
        return candidate_features, ranking, curve

    rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
    rfe.fit(xs, y)
    ranking = pd.DataFrame({"feature": candidate_features, "ranking": rfe.ranking_}).sort_values(["ranking", "feature"])
    ordered_features = ranking["feature"].tolist()

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for n_features in range(1, len(ordered_features) + 1):
        keep = ordered_features[:n_features]
        keep_idx = [candidate_features.index(f) for f in keep]
        scores = cross_val_score(estimator, xs[:, keep_idx], y, cv=cv, scoring="roc_auc")
        rows.append(
            {
                "n_features": n_features,
                "mean_auc": float(np.mean(scores)),
                "std_auc": float(np.std(scores)),
                "se_auc": float(np.std(scores) / np.sqrt(cv_splits)),
            }
        )
    curve = pd.DataFrame(rows)
    best_idx = int(curve["mean_auc"].idxmax())
    best_mean = float(curve.loc[best_idx, "mean_auc"])
    best_se = float(curve.loc[best_idx, "se_auc"])
    best_mean = float(curve.loc[best_idx, "mean_auc"])
    best_se = float(curve.loc[best_idx, "se_auc"])
    eligible = curve[curve["mean_auc"] >= best_mean - best_se].sort_values("n_features")
    selected_n = int(eligible.iloc[0]["n_features"])
    curve["selected"] = curve["n_features"] == selected_n
    return ordered_features[:selected_n], ranking, curve


def lasso_select(
    x: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int,
    mrmr_top_k: int,
) -> FeatureSelectionResult:
    x = remove_low_information_features(x)
    if x.empty:
        return FeatureSelectionResult(
            [],
            [],
            [],
            pd.DataFrame(columns=["feature", "coef_abs", "coef"]),
            pd.DataFrame(columns=["C", "lambda", "mean_mse", "std_mse", "se_mse", "selected_lambda_1se"]),
            pd.DataFrame(columns=["n_features", "mean_auc", "std_auc", "se_auc", "selected"]),
            pd.DataFrame(columns=["feature", "ranking"]),
        )

    mrmr_features = mrmr_select(x, y, top_k=min(mrmr_top_k, x.shape[1]))
    x_mrmr = x.loc[:, mrmr_features]
    xs, _, _ = impute_and_scale_fit(x_mrmr)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 40),
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring="neg_mean_squared_error",
        class_weight="balanced",
        max_iter=8000,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
    )
    clf.fit(xs, y)

    scores = clf.scores_[1] if 1 in clf.scores_ else next(iter(clf.scores_.values()))
    mse = -scores
    mean_mse = mse.mean(axis=0)
    std_mse = mse.std(axis=0)
    se_mse = std_mse / np.sqrt(cv_splits)
    best_idx = int(np.argmin(mean_mse))
    mse_threshold = float(mean_mse[best_idx] + se_mse[best_idx])
    eligible = np.where(mean_mse <= mse_threshold)[0]
    selected_idx = int(eligible[0])
    selected_c = float(clf.Cs_[selected_idx])
    lasso_cv = pd.DataFrame(
        {
            "C": clf.Cs_,
            "lambda": 1.0 / clf.Cs_,
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "se_mse": se_mse,
            "selected_lambda_1se": np.arange(len(clf.Cs_)) == selected_idx,
        }
    )

    final_lasso = LogisticRegression(
        C=selected_c,
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=8000,
        random_state=RANDOM_STATE,
    )
    final_lasso.fit(xs, y)
    coef = final_lasso.coef_.ravel()
    coef_df = pd.DataFrame({"feature": mrmr_features, "coef": coef, "coef_abs": np.abs(coef)})
    coef_df = coef_df.sort_values("coef_abs", ascending=False)
    lasso_nonzero = coef_df.loc[coef_df["coef_abs"] > 1e-8, "feature"].tolist()
    selected, rfe_ranking, rfe_curve = rfe_one_se_select(x_mrmr, y, lasso_nonzero, cv_splits=cv_splits)
    return FeatureSelectionResult(selected, mrmr_features, lasso_nonzero, coef_df, lasso_cv, rfe_curve, rfe_ranking)


def logistic_regression_table(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    x = remove_low_information_features(x, max_missing_rate=0.30)
    if x.empty:
        return pd.DataFrame(columns=["feature", "coef", "odds_ratio", "p_value"])
    xs, _, _ = impute_and_scale_fit(x)
    design = sm.add_constant(xs, has_constant="add")
    try:
        model = sm.Logit(y, design).fit(disp=False, maxiter=300)
    except Exception:
        return pd.DataFrame(
            {
                "feature": x.columns,
                "coef": np.nan,
                "odds_ratio": np.nan,
                "p_value": np.nan,
            }
        )
    return pd.DataFrame(
        {
            "feature": x.columns,
            "coef": model.params[1:],
            "odds_ratio": np.exp(model.params[1:]),
            "p_value": model.pvalues[1:],
        }
    ).sort_values("p_value", na_position="last")


def format_continuous(values: pd.Series, use_parametric: bool) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    if use_parametric:
        return f"{values.mean():.2f} ± {values.std():.2f}"
    return f"{values.median():.2f} ({values.quantile(0.25):.2f}, {values.quantile(0.75):.2f})"


def format_categorical(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    count = int((values == 1).sum())
    return f"{count}/{len(values)} ({count / len(values) * 100:.1f}%)"


def infer_clinical_variable_type(values: pd.Series) -> str:
    unique = set(values.dropna().unique().tolist())
    return "categorical" if unique.issubset({0, 1}) else "continuous"


def univariate_clinical_test(train_df: pd.DataFrame, feature: str, y: np.ndarray) -> dict[str, object]:
    values = pd.to_numeric(train_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    group0 = values[y == 0].dropna()
    group1 = values[y == 1].dropna()
    variable_type = infer_clinical_variable_type(values)
    if variable_type == "categorical":
        table = pd.crosstab(pd.Series(y, name="mRS_binary"), values.fillna(-999), dropna=False)
        if -999 in table.columns:
            table = table.drop(columns=[-999])
        if table.shape == (2, 2):
            expected = stats.contingency.expected_freq(table)
            if (expected < 5).any():
                _, p_value = stats.fisher_exact(table)
                method = "Fisher exact test"
            else:
                _, p_value, _, _ = stats.chi2_contingency(table)
                method = "Chi-square test"
        else:
            _, p_value, _, _ = stats.chi2_contingency(table)
            method = "Chi-square test"
        desc0 = format_categorical(group0)
        desc1 = format_categorical(group1)
    else:
        normal0 = len(group0) >= 3 and stats.shapiro(group0.sample(min(len(group0), 500), random_state=RANDOM_STATE)).pvalue > 0.05
        normal1 = len(group1) >= 3 and stats.shapiro(group1.sample(min(len(group1), 500), random_state=RANDOM_STATE)).pvalue > 0.05
        if normal0 and normal1:
            equal_var = stats.levene(group0, group1).pvalue > 0.05
            _, p_value = stats.ttest_ind(group0, group1, equal_var=equal_var, nan_policy="omit")
            method = "Student t-test" if equal_var else "Welch t-test"
            use_parametric = True
        else:
            _, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
            method = "Mann-Whitney U test"
            use_parametric = False
        desc0 = format_continuous(group0, use_parametric)
        desc1 = format_continuous(group1, use_parametric)
    return {
        "feature": feature,
        "variable_type": variable_type,
        "method": method,
        "mRS_0_2": desc0,
        "mRS_3_6": desc1,
        "p_value": p_value,
    }


def select_clinical_features(train_df: pd.DataFrame, clinical_cols: list[str], y: np.ndarray) -> ClinicalSelectionResult:
    x_clinical = prepare_numeric_matrix(train_df, clinical_cols)
    x_clinical = remove_low_information_features(x_clinical, max_missing_rate=0.30)
    univariate_rows = [univariate_clinical_test(train_df, col, y) for col in x_clinical.columns]
    univariate = pd.DataFrame(univariate_rows).sort_values("p_value", na_position="last")
    candidates = univariate.loc[univariate["p_value"] < UNIVARIATE_P_THRESHOLD, "feature"].tolist()
    if not candidates:
        candidates = univariate.head(1)["feature"].tolist()

    current = candidates.copy()
    while len(current) > 1:
        multi = logistic_regression_table(x_clinical[current], y)
        if multi["p_value"].isna().all():
            break
        worst = multi.sort_values("p_value", ascending=False, na_position="first").iloc[0]
        if pd.notna(worst["p_value"]) and worst["p_value"] <= MULTIVARIATE_P_THRESHOLD:
            break
        current.remove(worst["feature"])

    multivariate = logistic_regression_table(x_clinical[current], y)
    independent = multivariate.loc[multivariate["p_value"] <= MULTIVARIATE_P_THRESHOLD, "feature"].tolist()
    if not independent:
        independent = multivariate.head(1)["feature"].tolist()
    selected, rfe_ranking, rfe_curve = rfe_one_se_select(x_clinical, y, independent, cv_splits=5)
    if not selected:
        selected = independent
    return ClinicalSelectionResult(selected, univariate, multivariate, rfe_curve, rfe_ranking)


def build_model_feature_sets(
    groups: dict[str, list[str]],
    train_df: pd.DataFrame,
    y: np.ndarray,
    clinical: list[str],
) -> tuple[dict[str, list[str]], dict[str, FeatureSelectionResult]]:
    radiomics_sets = {
        "M2": groups["pre"],
        "M3": groups["post"],
        "M4": groups["delta"],
    }
    selection: dict[str, FeatureSelectionResult] = {}
    feature_sets = {"M1": clinical}
    for model_name, cols in radiomics_sets.items():
        candidate_cols = list(dict.fromkeys(cols))
        if not candidate_cols:
            raise ValueError(f"{model_name} has no available radiomics candidate features.")
        x_candidate = prepare_numeric_matrix(train_df, candidate_cols)
        selection[model_name] = lasso_select(
            x_candidate,
            y,
            cv_splits=5,
            mrmr_top_k=MRMR_TOP_K,
        )
        feature_sets[model_name] = selection[model_name].selected_radiomics
    return feature_sets, selection


def make_xgb(y: np.ndarray) -> XGBClassifier:
    negatives = max(1, int(np.sum(y == 0)))
    positives = max(1, int(np.sum(y == 1)))
    return XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=2.0,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=negatives / positives,
    )


def tune_xgb(x: np.ndarray, y: np.ndarray) -> XGBClassifier:
    base = make_xgb(y)
    param_distributions = {
        "n_estimators": [250, 350, 450, 600],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.015, 0.025, 0.04, 0.06],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.65, 0.8, 0.95],
        "min_child_weight": [1, 3, 5, 8],
        "reg_alpha": [0.0, 0.05, 0.15, 0.3],
        "reg_lambda": [1.0, 2.0, 4.0, 8.0],
    }
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=XGB_RANDOM_SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(x, y)
    best_model = search.best_estimator_
    setattr(best_model, "_tuning_best_score", float(search.best_score_))
    setattr(best_model, "_tuning_best_params", search.best_params_)
    return best_model


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        threshold = 0.5
    return threshold


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def auc_ci_delong(y_true: np.ndarray, y_prob: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives < 2 or negatives < 2:
        auc_value = roc_auc_score(y_true, y_prob) if positives and negatives else np.nan
        return float(auc_value), np.nan, np.nan

    order = np.argsort(-y_true)
    aucs, cov = fast_delong(y_prob[np.newaxis, order], positives)
    auc_value = float(aucs[0])
    se = math.sqrt(max(float(cov[0, 0]), 0.0)) if cov.size else np.nan
    if not np.isfinite(se):
        return auc_value, np.nan, np.nan
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return auc_value, max(0.0, auc_value - z * se), min(1.0, auc_value + z * se)


def proportion_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if total <= 0:
        return np.nan, np.nan, np.nan
    p = successes / total
    z = NormalDist().inv_cdf(1 - alpha / 2)
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half_width = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return float(p), max(0.0, center - half_width), min(1.0, center + half_width)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true == 1))
    if label_1_count < 2 or len(y_true) - label_1_count < 2:
        return np.nan
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return np.nan
    z = abs(diff) / math.sqrt(var)
    return 2 * (1 - NormalDist().cdf(z))


def compute_performance_ci_table(
    perf: pd.DataFrame,
    probabilities: dict[tuple[str, str], np.ndarray],
    datasets: dict[str, pd.DataFrame],
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in perf.iterrows():
        dataset_name = row["Dataset"]
        model_name = row["Model"]
        if (dataset_name, model_name) not in probabilities:
            continue
        if dataset_name == "Train_CV":
            if train_df is None:
                continue
            y = train_df["mRS_binary"].to_numpy()
        else:
            y = datasets[dataset_name]["mRS_binary"].to_numpy()
        prob = probabilities[(dataset_name, model_name)]
        auc_value, auc_low, auc_high = auc_ci_delong(y, prob)
        sens, sens_low, sens_high = proportion_ci(int(row["TP"]), int(row["TP"] + row["FN"]))
        spec, spec_low, spec_high = proportion_ci(int(row["TN"]), int(row["TN"] + row["FP"]))
        rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "N": int(row["N"]),
                "AUC": auc_value,
                "AUC_95CI_low": auc_low,
                "AUC_95CI_high": auc_high,
                "AUC_95CI": format_ci(auc_value, auc_low, auc_high),
                "Sensitivity": sens,
                "Sensitivity_95CI_low": sens_low,
                "Sensitivity_95CI_high": sens_high,
                "Sensitivity_95CI": format_ci(sens, sens_low, sens_high),
                "Specificity": spec,
                "Specificity_95CI_low": spec_low,
                "Specificity_95CI_high": spec_high,
                "Specificity_95CI": format_ci(spec, spec_low, spec_high),
                "Threshold": float(row["Threshold"]),
                "TN": int(row["TN"]),
                "FP": int(row["FP"]),
                "FN": int(row["FN"]),
                "TP": int(row["TP"]),
            }
        )
    return pd.DataFrame(rows)


def format_ci(value: float, low: float, high: float) -> str:
    if not np.isfinite(value) or not np.isfinite(low) or not np.isfinite(high):
        return ""
    return f"{value:.3f} ({low:.3f}, {high:.3f})"


def compute_test_center_delong_table(
    probabilities: dict[tuple[str, str], np.ndarray],
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    model_names = BASE_MODEL_NAMES + FUSION_MODEL_NAMES
    for dataset_name, df in datasets.items():
        if df["dataset_role"].iloc[0] != "test":
            continue
        y = df["mRS_binary"].to_numpy()
        available_models = [model_name for model_name in model_names if (dataset_name, model_name) in probabilities]
        for model_a, model_b in combinations(available_models, 2):
            prob_a = probabilities[(dataset_name, model_a)]
            prob_b = probabilities[(dataset_name, model_b)]
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "AUC_A": roc_auc_score(y, prob_a),
                    "AUC_B": roc_auc_score(y, prob_b),
                    "AUC_diff_A_minus_B": roc_auc_score(y, prob_a) - roc_auc_score(y, prob_b),
                    "Delong_p_value": delong_pvalue(y, prob_a, prob_b),
                }
            )
    return pd.DataFrame(rows)


def hosmer_lemeshow_test(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    data = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "prob": np.asarray(y_prob, dtype=float)})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data["y"].nunique() < 2:
        return {"HL_chi2": np.nan, "HL_df": np.nan, "HL_p_value": np.nan, "HL_bins": 0}

    bins = min(n_bins, len(data))
    try:
        data["bin"] = pd.qcut(data["prob"], q=bins, duplicates="drop")
    except ValueError:
        data["bin"] = pd.cut(data["prob"], bins=bins)
    grouped = data.groupby("bin", observed=False)
    obs = grouped["y"].sum()
    total = grouped["y"].count()
    exp = grouped["prob"].sum()
    obs_nonevent = total - obs
    exp_nonevent = total - exp
    valid = (exp > 0) & (exp_nonevent > 0)
    groups = int(valid.sum())
    if groups < 3:
        return {"HL_chi2": np.nan, "HL_df": np.nan, "HL_p_value": np.nan, "HL_bins": groups}
    chi2 = (((obs[valid] - exp[valid]) ** 2 / exp[valid]) + ((obs_nonevent[valid] - exp_nonevent[valid]) ** 2 / exp_nonevent[valid])).sum()
    df = groups - 2
    return {"HL_chi2": float(chi2), "HL_df": int(df), "HL_p_value": float(stats.chi2.sf(chi2, df)), "HL_bins": groups}


def compute_hl_calibration_table(
    probabilities: dict[tuple[str, str], np.ndarray],
    datasets: dict[str, pd.DataFrame],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    model_names = BASE_MODEL_NAMES + FUSION_MODEL_NAMES
    for dataset_name, df in datasets.items():
        if df["dataset_role"].iloc[0] != "test":
            continue
        y = df["mRS_binary"].to_numpy()
        for model_name in model_names:
            key = (dataset_name, model_name)
            if key not in probabilities:
                continue
            result = hosmer_lemeshow_test(y, probabilities[key], n_bins=n_bins)
            rows.append({"Dataset": dataset_name, "Model": model_name, **result})
    return pd.DataFrame(rows)


def fit_and_predict(
    train_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, XGBClassifier], dict[tuple[str, str], np.ndarray], dict[str, dict[str, object]]]:
    y_train = train_df["mRS_binary"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    performance_rows = []
    thresholds_rows = []
    fitted_models = {}
    probabilities: dict[tuple[str, str], np.ndarray] = {}
    tuning_rows = []

    for model_name, cols in feature_sets.items():
        x_train = prepare_numeric_matrix(train_df, cols)
        x_train = remove_low_information_features(x_train)
        used_cols = x_train.columns.tolist()
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x_train)
        model = tune_xgb(x_imp, y_train)
        cv_prob = cross_val_predict(clone(model), x_imp, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = youden_threshold(y_train, cv_prob)
        probabilities[("Train_CV", model_name)] = cv_prob
        row = metric_row(y_train, cv_prob, threshold)
        row.update({"Dataset": "Train_CV", "Model": model_name, "N": len(y_train), "P_vs_M1": np.nan})
        performance_rows.append(row)
        thresholds_rows.append({"Dataset": "Train_CV", "Model": model_name, **row})

        model.fit(x_imp, y_train)
        fitted_models[model_name] = model
        setattr(model, "_analysis_columns", used_cols)
        setattr(model, "_analysis_imputer", imputer)
        setattr(model, "_analysis_threshold", threshold)
        tuning_rows.append(
            {
                "Model": model_name,
                "Best_CV_AUC": getattr(model, "_tuning_best_score", np.nan),
                **getattr(model, "_tuning_best_params", {}),
            }
        )

        for dataset_name, df in datasets.items():
            if df["dataset_role"].iloc[0] != "test":
                continue
            y = df["mRS_binary"].to_numpy()
            x = prepare_numeric_matrix(df, used_cols)
            prob = model.predict_proba(imputer.transform(x))[:, 1]
            probabilities[(dataset_name, model_name)] = prob
            test_row = metric_row(y, prob, threshold)
            test_row.update({"Dataset": dataset_name, "Model": model_name, "N": len(y), "P_vs_M1": np.nan})
            performance_rows.append(test_row)
            thresholds_rows.append({"Dataset": dataset_name, "Model": model_name, **test_row})

    perf = pd.DataFrame(performance_rows)
    for dataset_name in perf["Dataset"].unique():
        y = y_train if dataset_name == "Train_CV" else datasets[dataset_name]["mRS_binary"].to_numpy()
        m1_prob = probabilities[(dataset_name, "M1")]
        for model_name in feature_sets:
            idx = (perf["Dataset"] == dataset_name) & (perf["Model"] == model_name)
            if model_name == "M1":
                perf.loc[idx, "P_vs_M1"] = np.nan
            else:
                perf.loc[idx, "P_vs_M1"] = delong_pvalue(y, probabilities[(dataset_name, model_name)], m1_prob)

    fusion_perf, fusion_thresholds, fusion_models = add_decision_fusion_results(
        train_df,
        datasets,
        probabilities,
        base_model_names=list(feature_sets),
    )
    perf = pd.concat([perf, fusion_perf], ignore_index=True)
    thresholds_rows.extend(fusion_thresholds.to_dict("records"))

    tuning_df = pd.DataFrame(tuning_rows)
    for model in fitted_models.values():
        setattr(model, "_analysis_tuning_table", tuning_df)
    return perf, pd.DataFrame(thresholds_rows), fitted_models, probabilities, fusion_models


def _probability_matrix(
    probabilities: dict[tuple[str, str], np.ndarray],
    dataset_name: str,
    model_names: list[str],
) -> np.ndarray:
    return np.column_stack([probabilities[(dataset_name, model_name)] for model_name in model_names])


def add_decision_fusion_results(
    train_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    probabilities: dict[tuple[str, str], np.ndarray],
    base_model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    base_model_names = base_model_names or BASE_MODEL_NAMES
    y_train = train_df["mRS_binary"].to_numpy()
    performance_rows = []
    thresholds_rows = []
    dataset_names = ["Train_CV"] + [k for k, v in datasets.items() if v["dataset_role"].iloc[0] == "test"]
    threshold_by_model = {}
    fusion_estimators = {}
    fusion_inputs = []
    fusion_models = {}

    for size in range(2, len(base_model_names) + 1):
        for combo in combinations(base_model_names, size):
            if not is_valid_fusion_combo(combo):
                continue
            combo = list(combo)
            model_name = f"Fusion_stack_{'_'.join(combo)}"
            train_matrix = _probability_matrix(probabilities, "Train_CV", combo)
            fusion_estimator = make_xgb(y_train)
            fusion_estimator.set_params(
                n_estimators=160,
                max_depth=2,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=1.0,
                reg_alpha=0.05,
                reg_lambda=2.5,
                min_child_weight=2,
            )
            fusion_estimator.fit(train_matrix, y_train)
            train_prob = fusion_estimator.predict_proba(train_matrix)[:, 1]
            fusion_estimators[model_name] = fusion_estimator
            fusion_inputs.append((model_name, combo))
            threshold_by_model[model_name] = youden_threshold(y_train, train_prob)
            probabilities[("Train_CV", model_name)] = train_prob
            fusion_models[model_name] = {
                "model": fusion_estimator,
                "inputs": combo,
                "threshold": threshold_by_model[model_name],
            }

    for dataset_name in dataset_names:
        y = y_train if dataset_name == "Train_CV" else datasets[dataset_name]["mRS_binary"].to_numpy()
        m1_prob = probabilities[(dataset_name, "M1")]
        for model_name, combo in fusion_inputs:
            matrix = _probability_matrix(probabilities, dataset_name, combo)
            prob = fusion_estimators[model_name].predict_proba(matrix)[:, 1]
            probabilities[(dataset_name, model_name)] = prob
            threshold = threshold_by_model[model_name]
            row = metric_row(y, prob, threshold)
            row.update(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "N": len(y),
                    "P_vs_M1": delong_pvalue(y, prob, m1_prob),
                }
            )
            performance_rows.append(row)
            thresholds_rows.append({"Dataset": dataset_name, "Model": model_name, **row})

    return pd.DataFrame(performance_rows), pd.DataFrame(thresholds_rows), fusion_models


def save_trained_model_bundles(
    output_dir: Path,
    fitted_models: dict[str, XGBClassifier],
    thresholds: pd.DataFrame,
    fusion_models: dict[str, dict[str, object]] | None = None,
) -> None:
    """Save trained model bundles for direct reuse."""
    model_dir = output_dir / "trained_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name, model in fitted_models.items():
        threshold_rows = thresholds[(thresholds["Dataset"] == "Train_CV") & (thresholds["Model"] == model_name)]
        threshold = float(threshold_rows.iloc[0]["Threshold"]) if not threshold_rows.empty else float(getattr(model, "_analysis_threshold", 0.5))
        columns = list(getattr(model, "_analysis_columns"))
        imputer = getattr(model, "_analysis_imputer")
        bundle = {
            "model_name": model_name,
            "model": model,
            "columns": columns,
            "imputer": imputer,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
        }
        path = model_dir / f"{model_name}.joblib"
        joblib.dump(bundle, path)
        rows.append(
            {
                "Model": model_name,
                "Type": "base",
                "Path": str(path),
                "Threshold": threshold,
                "Feature_count": len(columns),
                "Input_models": "",
            }
        )

    fusion_models = fusion_models or {}
    final_fusion = fusion_models.get(FINAL_FUSION_MODEL_NAME)
    if final_fusion is not None:
        fusion_threshold_rows = thresholds[
            (thresholds["Dataset"] == "Train_CV")
            & (thresholds["Model"] == FINAL_FUSION_MODEL_NAME)
        ]
        threshold = (
            float(fusion_threshold_rows.iloc[0]["Threshold"])
            if not fusion_threshold_rows.empty
            else float(final_fusion.get("threshold", 0.5))
        )
        input_models = list(final_fusion["inputs"])
        bundle = {
            "model_name": FINAL_FUSION_MODEL_NAME,
            "model_type": "stacking_fusion",
            "model": final_fusion["model"],
            "input_models": input_models,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
        }
        path = model_dir / f"{FINAL_FUSION_MODEL_NAME}.joblib"
        joblib.dump(bundle, path)
        rows.append(
            {
                "Model": FINAL_FUSION_MODEL_NAME,
                "Type": "stacking_fusion",
                "Path": str(path),
                "Threshold": threshold,
                "Feature_count": len(input_models),
                "Input_models": "+".join(input_models),
            }
        )

    pd.DataFrame(rows).to_csv(model_dir / "manifest.csv", index=False, encoding="utf-8-sig")


def load_trained_model_bundles(model_dir: Path) -> dict[str, dict[str, object]]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Trained model directory not found: {model_dir}. Run the full pipeline first.")

    bundles = {}
    for model_name in BASE_MODEL_NAMES:
        path = model_dir / f"{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model file: {path}. Run the full pipeline first.")
        bundle = joblib.load(path)
        bundles[model_name] = bundle

    return bundles


def predict_with_trained_model_bundles(
    datasets: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray], dict[str, dict[str, object]]]:
    performance_rows = []
    thresholds_rows = []
    probabilities: dict[tuple[str, str], np.ndarray] = {}

    for dataset_name, df in datasets.items():
        y = df["mRS_binary"].to_numpy()
        for model_name, bundle in bundles.items():
            columns = list(bundle["columns"])
            imputer = bundle["imputer"]
            model = bundle["model"]
            threshold = float(bundle.get("threshold", 0.5))
            x = prepare_numeric_matrix(df, columns)
            prob = model.predict_proba(imputer.transform(x))[:, 1]
            probabilities[(dataset_name, model_name)] = prob
            if df["dataset_role"].iloc[0] != "test":
                continue
            row = metric_row(y, prob, threshold)
            row.update({"Dataset": dataset_name, "Model": model_name, "N": len(y), "P_vs_M1": np.nan})
            performance_rows.append(row)
            thresholds_rows.append({"Dataset": dataset_name, "Model": model_name, **row})

    perf = pd.DataFrame(performance_rows)
    for dataset_name in perf["Dataset"].unique():
        y = datasets[dataset_name]["mRS_binary"].to_numpy()
        m1_prob = probabilities[(dataset_name, "M1")]
        for model_name in bundles:
            idx = (perf["Dataset"] == dataset_name) & (perf["Model"] == model_name)
            if model_name == "M1":
                perf.loc[idx, "P_vs_M1"] = np.nan
            else:
                perf.loc[idx, "P_vs_M1"] = delong_pvalue(y, probabilities[(dataset_name, model_name)], m1_prob)

    train_names = [name for name, df in datasets.items() if df["dataset_role"].iloc[0] == "train"]
    if not train_names:
        raise ValueError("Test-only mode requires at least one training set to build decision fusion models.")
    train_df = pd.concat([datasets[name] for name in train_names], ignore_index=True)
    for model_name in bundles:
        probabilities[("Train_CV", model_name)] = np.concatenate(
            [probabilities[(name, model_name)] for name in train_names]
        )
    fusion_perf, fusion_thresholds, fusion_models = add_decision_fusion_results(
        train_df,
        datasets,
        probabilities,
        base_model_names=list(bundles),
    )
    fusion_perf = fusion_perf[fusion_perf["Dataset"] != "Train_CV"].copy()
    fusion_thresholds = fusion_thresholds[fusion_thresholds["Dataset"] != "Train_CV"].copy()
    perf = pd.concat([perf, fusion_perf], ignore_index=True)
    thresholds_rows.extend(fusion_thresholds.to_dict("records"))

    return perf, pd.DataFrame(thresholds_rows), probabilities, fusion_models


def save_prediction_probabilities(
    datasets: dict[str, pd.DataFrame],
    probabilities: dict[tuple[str, str], np.ndarray],
    output_path: Path,
) -> None:
    rows = []
    model_names = sorted({model_name for dataset_name, model_name in probabilities if dataset_name in datasets})

    for dataset_name, df in datasets.items():
        if df["dataset_role"].iloc[0] != "test":
            continue
        patient_ids = df["patient_id"].astype(str).tolist() if "patient_id" in df.columns else [str(i) for i in range(len(df))]
        for idx, patient_id in enumerate(patient_ids):
            row = {
                "Dataset": dataset_name,
                "patient_id": patient_id,
                "mRS": int(df["mRS"].iloc[idx]),
                "mRS_binary": int(df["mRS_binary"].iloc[idx]),
            }
            for model_name in model_names:
                key = (dataset_name, model_name)
                if key in probabilities:
                    row[f"{model_name}_prob"] = float(probabilities[key][idx])
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def baseline_table(train_df: pd.DataFrame, datasets: dict[str, pd.DataFrame], clinical_cols: list[str]) -> pd.DataFrame:
    cohorts = {"Train": train_df}
    cohorts.update({k: v for k, v in datasets.items() if v["dataset_role"].iloc[0] == "test"})
    rows = []
    for col in ["mRS_binary"] + clinical_cols:
        row = {"Variable": col}
        train_values = pd.to_numeric(train_df[col], errors="coerce")
        for name, df in cohorts.items():
            values = pd.to_numeric(df[col], errors="coerce")
            nonnull = values.dropna()
            unique_vals = set(nonnull.unique().tolist())
            if unique_vals.issubset({0, 1}):
                count = int((nonnull == 1).sum())
                pct = count / len(nonnull) * 100 if len(nonnull) else 0
                row[name] = f"{count}/{len(nonnull)} ({pct:.1f}%)"
            else:
                row[name] = f"{nonnull.mean():.2f} ± {nonnull.std():.2f}"
        for name, df in cohorts.items():
            if name == "Train":
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            train_nonnull = train_values.dropna()
            values_nonnull = values.dropna()
            if set(train_nonnull.unique()).issubset({0, 1}) and set(values_nonnull.unique()).issubset({0, 1}):
                table = pd.crosstab(
                    pd.Series(["train"] * len(train_nonnull) + [name] * len(values_nonnull)),
                    pd.Series(train_nonnull.tolist() + values_nonnull.tolist()),
                )
                p_value = stats.chi2_contingency(table)[1] if table.shape == (2, 2) else np.nan
            else:
                p_value = stats.mannwhitneyu(train_nonnull, values_nonnull, alternative="two-sided").pvalue
            row[f"P_Train_vs_{name}"] = p_value
        rows.append(row)
    return pd.DataFrame(rows)


def feature_count_table(feature_sets: dict[str, list[str]], groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for model_name, cols in feature_sets.items():
        colset = set(cols)
        clinical = len(colset.intersection(groups["clinical"]))
        pre = len(colset.intersection(groups["pre"]))
        post = len(colset.intersection(groups["post"]))
        delta = len(colset.intersection(groups["delta"]))
        rows.append(
            {
                "Model": model_name,
                "Clinical_features": clinical,
                "Pre_radiomics": pre,
                "Post_radiomics": post,
                "Delta_features": delta,
                "Total_features": len(cols),
            }
        )
    return pd.DataFrame(rows)


def plot_lasso(selection: FeatureSelectionResult, train_df: pd.DataFrame, y: np.ndarray, output_dir: Path, model_name: str) -> None:
    features = selection.mrmr_features[: min(80, len(selection.mrmr_features))]
    x = prepare_numeric_matrix(train_df, features)
    xs, _, _ = impute_and_scale_fit(x)
    alphas, coefs, _ = lasso_path(xs, y.astype(float), eps=1e-3, n_alphas=80)

    plt.figure(figsize=(8, 6))
    for coef in coefs:
        plt.plot(np.log10(alphas), coef, linewidth=0.8, alpha=0.7)
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient")
    plt.title("LASSO coefficient path")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_lasso_path_{model_name}.png", dpi=300)
    plt.close()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cs = np.logspace(-3, 2, 40)
    rows = []
    for c in cs:
        fold_mse = []
        for tr, va in cv.split(xs, y):
            clf = LogisticRegression(
                C=c,
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )
            clf.fit(xs[tr], y[tr])
            prob = clf.predict_proba(xs[va])[:, 1]
            fold_mse.append(mean_squared_error(y[va], prob))
        rows.append({"log_lambda": np.log10(1 / c), "mean_mse": np.mean(fold_mse), "std_mse": np.std(fold_mse)})
    mse_df = pd.DataFrame(rows).sort_values("log_lambda")
    mse_df.to_csv(output_dir / f"lasso_cv_mse_{model_name}.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(7, 5))
    plt.errorbar(mse_df["log_lambda"], mse_df["mean_mse"], yerr=mse_df["std_mse"], fmt="-o", markersize=3, capsize=2)
    plt.xlabel("log10(lambda)")
    plt.ylabel("5-fold CV MSE")
    plt.title("LASSO 5-fold CV MSE")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_lasso_cv_mse_{model_name}.png", dpi=300)
    plt.close()


def plot_rfe_curve(curve: pd.DataFrame, title: str, output_path: Path) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.errorbar(curve["n_features"], curve["mean_auc"], yerr=curve["se_auc"], fmt="-o", markersize=4, capsize=2)
    selected = curve[curve["selected"]]
    if not selected.empty:
        plt.axvline(int(selected.iloc[0]["n_features"]), color="red", linestyle="--", linewidth=1)
    plt.xlabel("Number of selected features")
    plt.ylabel("5-fold CV AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc(perf_probs: dict[tuple[str, str], np.ndarray], datasets: dict[str, pd.DataFrame], train_df: pd.DataFrame, output_dir: Path) -> None:
    for dataset_name in ["Train_CV"] + [k for k, v in datasets.items() if v["dataset_role"].iloc[0] == "test"]:
        y = train_df["mRS_binary"].to_numpy() if dataset_name == "Train_CV" else datasets[dataset_name]["mRS_binary"].to_numpy()
        plt.figure(figsize=(7, 6))
        for model_name in BASE_MODEL_NAMES + FUSION_MODEL_NAMES:
            prob = perf_probs[(dataset_name, model_name)]
            fpr, tpr, _ = roc_curve(y, prob)
            plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC={roc_auc_score(y, prob):.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"ROC curves ({dataset_name})")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_roc_{dataset_name}.png", dpi=300)
        plt.close()


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y_true)
    out = []
    for pt in thresholds:
        pred = y_prob >= pt
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        out.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.asarray(out)


def plot_dca_and_calibration(perf_probs: dict[tuple[str, str], np.ndarray], datasets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for dataset_name, df in datasets.items():
        if df["dataset_role"].iloc[0] != "test":
            continue
        y = df["mRS_binary"].to_numpy()
        thresholds = np.linspace(0.01, 0.99, 99)
        plt.figure(figsize=(7, 5))
        dca_curves = []
        for model_name in BASE_MODEL_NAMES + FUSION_MODEL_NAMES:
            prob = perf_probs[(dataset_name, model_name)]
            nb = net_benefit(y, prob, thresholds)
            dca_curves.append(nb)
            plt.plot(thresholds, nb, linewidth=2, label=model_name)
        treat_all = np.mean(y) - (1 - np.mean(y)) * thresholds / (1 - thresholds)
        plt.plot(thresholds, treat_all, "k--", linewidth=1, label="Treat all")
        plt.plot(thresholds, np.zeros_like(thresholds), "k:", linewidth=1, label="Treat none")
        plt.xlabel("Threshold probability")
        plt.ylabel("Net benefit")
        plt.title(f"Decision curve analysis ({dataset_name})")
        visible_curves = dca_curves + [np.zeros_like(thresholds), treat_all[thresholds <= 0.80]]
        visible_max = max(float(np.nanmax(curve)) for curve in visible_curves)
        plt.ylim(-0.05, max(0.10, visible_max + 0.05))
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_dca_all_models_fusion_{dataset_name}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 6))
        for model_name in BASE_MODEL_NAMES + FUSION_MODEL_NAMES:
            prob = perf_probs[(dataset_name, model_name)]
            frac_pos, mean_pred = calibration_curve(y, prob, n_bins=8, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=model_name)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed probability")
        plt.title(f"Calibration curves ({dataset_name})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_calibration_{dataset_name}.png", dpi=300)
        plt.close()


def plot_confusion_matrices(thresholds: pd.DataFrame, output_dir: Path) -> None:
    test_thresholds = thresholds[thresholds["Dataset"] != "Train_CV"].copy()
    for dataset_name in test_thresholds["Dataset"].unique():
        subset = test_thresholds[test_thresholds["Dataset"] == dataset_name]
        best_model = subset.sort_values("AUC", ascending=False).iloc[0]["Model"]
        for model_name in subset["Model"].unique():
            row = subset[subset["Model"] == model_name].iloc[0]
            matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
            plt.figure(figsize=(4.5, 4))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred 0-2", "Pred 3-6"], yticklabels=["True 0-2", "True 3-6"])
            plt.title(f"{dataset_name} {model_name}")
            plt.tight_layout()
            suffix = "best" if model_name == best_model else "supplement"
            plt.savefig(output_dir / f"fig_confusion_{dataset_name}_{model_name}_{suffix}.png", dpi=300)
            plt.close()


def explain_all_models(
    train_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    perf: pd.DataFrame,
    models: dict[str, XGBClassifier],
    output_dir: Path,
) -> None:
    y_train = train_df["mRS_binary"].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    meta = {
        "best_model_by_train_cv_auc": perf[perf["Dataset"] == "Train_CV"].sort_values("AUC", ascending=False).iloc[0]["Model"],
        "models": {},
    }
    for model_name in sorted(models):
        model = models[model_name]
        cols = getattr(model, "_analysis_columns")
        imputer = getattr(model, "_analysis_imputer")
        x_train = prepare_numeric_matrix(train_df, cols)
        x_imp = imputer.transform(x_train)

        impurity = pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        impurity.to_csv(output_dir / f"impurity_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(8, 6))
        sns.barplot(data=impurity.head(20), y="feature", x="importance", color="#4C78A8")
        plt.title(f"Mean impurity decrease ({model_name})")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_impurity_importance_{model_name}.png", dpi=300)
        plt.close()

        perm = permutation_importance(model, x_imp, y_train, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc")
        perm_df = pd.DataFrame({"feature": cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
        perm_df = perm_df.sort_values("importance_mean", ascending=False)
        perm_df.to_csv(output_dir / f"permutation_importance_{model_name}.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(8, 6))
        sns.barplot(data=perm_df.head(20), y="feature", x="importance_mean", color="#59A14F")
        plt.title(f"Permutation importance ({model_name})")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_permutation_importance_{model_name}.png", dpi=300)
        plt.close()

        sample_n = min(400, x_imp.shape[0])
        sample_idx = rng.choice(np.arange(x_imp.shape[0]), size=sample_n, replace=False)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_imp[sample_idx])
        x_sample_df = pd.DataFrame(x_imp[sample_idx], columns=cols)
        shap.summary_plot(shap_values, x_sample_df, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_shap_summary_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        patient_idx = int(sample_idx[0])
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            pd.DataFrame(x_imp[[patient_idx]], columns=cols).iloc[0],
            matplotlib=True,
            show=False,
        )
        plt.savefig(output_dir / f"fig_shap_force_patient_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        meta["models"][model_name] = {
            "force_plot_patient_id": str(train_df.iloc[patient_idx].get("patient_id", patient_idx)),
            "n_features": len(cols),
        }
    meta = {
        **meta,
    }
    (output_dir / "explainability_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def explain_full_stacking_fusion(
    train_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    probabilities: dict[tuple[str, str], np.ndarray],
    fusion_models: dict[str, dict[str, object]],
    output_dir: Path,
    model_names: list[str] | None = None,
) -> None:
    if model_names is None:
        model_names = list(fusion_models)[:2]
    if "Fusion_stack_M1_M2_M3_M4" in fusion_models and "Fusion_stack_M1_M2_M3_M4" not in model_names:
        model_names = list(model_names) + ["Fusion_stack_M1_M2_M3_M4"]
    for model_name in model_names:
        explain_one_stacking_fusion(train_df, datasets, probabilities, fusion_models, output_dir, model_name)


def best_fusion_model_names(perf: pd.DataFrame, top_n: int = 2) -> list[str]:
    fusion_perf = perf[perf["Model"].astype(str).str.startswith("Fusion_stack_")].copy()
    if fusion_perf.empty:
        return []

    test_perf = fusion_perf[fusion_perf["Dataset"].astype(str) != "Train_CV"].copy()
    source = test_perf if not test_perf.empty else fusion_perf
    ranked = (
        source.groupby("Model", as_index=False)["AUC"]
        .mean()
        .sort_values(["AUC", "Model"], ascending=[False, True])
    )
    return ranked.head(top_n)["Model"].astype(str).tolist()


def explain_one_stacking_fusion(
    train_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    probabilities: dict[tuple[str, str], np.ndarray],
    fusion_models: dict[str, dict[str, object]],
    output_dir: Path,
    model_name: str,
) -> None:
    if model_name not in fusion_models:
        return

    model_info = fusion_models[model_name]
    fusion_estimator = model_info["model"]
    input_models = list(model_info["inputs"])
    feature_names = [f"{name}_prob" for name in input_models]
    train_matrix = _probability_matrix(probabilities, "Train_CV", input_models)
    train_prob = probabilities[("Train_CV", model_name)]
    x_train = pd.DataFrame(train_matrix, columns=feature_names)

    gain_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": fusion_estimator.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    gain_df.to_csv(output_dir / f"xgboost_fusion_importance_{model_name}.csv", index=False, encoding="utf-8-sig")

    meta_df = pd.DataFrame(
        [{"Model": model_name, "Threshold": float(model_info["threshold"]), "Fusion_estimator": "XGBoost"}]
    )
    meta_df.to_csv(output_dir / f"xgboost_fusion_meta_{model_name}.csv", index=False, encoding="utf-8-sig")

    input_rows = []
    train_ids = train_df["patient_id"].tolist() if "patient_id" in train_df.columns else list(range(len(train_df)))
    for idx, patient_id in enumerate(train_ids):
        row = {
            "Dataset": "Train_CV",
            "patient_id": patient_id,
            "mRS_binary": int(train_df["mRS_binary"].iloc[idx]),
            f"{model_name}_prob": float(train_prob[idx]),
        }
        for col_idx, feature_name in enumerate(feature_names):
            row[feature_name] = float(train_matrix[idx, col_idx])
        input_rows.append(row)

    for dataset_name, df in datasets.items():
        if df["dataset_role"].iloc[0] != "test":
            continue
        matrix = _probability_matrix(probabilities, dataset_name, input_models)
        fusion_prob = probabilities[(dataset_name, model_name)]
        patient_ids = df["patient_id"].tolist() if "patient_id" in df.columns else list(range(len(df)))
        for idx, patient_id in enumerate(patient_ids):
            row = {
                "Dataset": dataset_name,
                "patient_id": patient_id,
                "mRS_binary": int(df["mRS_binary"].iloc[idx]),
                f"{model_name}_prob": float(fusion_prob[idx]),
            }
            for col_idx, feature_name in enumerate(feature_names):
                row[feature_name] = float(matrix[idx, col_idx])
            input_rows.append(row)

    pd.DataFrame(input_rows).to_csv(output_dir / f"stacking_input_probabilities_{model_name}.csv", index=False, encoding="utf-8-sig")

    perm = permutation_importance(
        fusion_estimator,
        train_matrix,
        train_df["mRS_binary"].to_numpy(),
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="roc_auc",
    )
    perm_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(output_dir / f"permutation_importance_{model_name}.csv", index=False, encoding="utf-8-sig")

    if model_name == "Fusion_stack_M1_M2_M3_M4":
        plt.figure(figsize=(7.5, 5.5))
        sns.barplot(data=gain_df, y="Feature", x="Importance", color="#4C78A8")
        plt.xlabel("Mean decrease impurity")
        plt.ylabel("Feature")
        plt.title(f"Mean decrease impurity ({model_name})")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_impurity_importance_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7.5, 5.5))
        sns.barplot(data=perm_df, y="Feature", x="importance_mean", color="#59A14F")
        plt.xlabel("Permutation importance")
        plt.ylabel("Feature")
        plt.title(f"Permutation importance ({model_name})")
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_permutation_importance_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    explainer = shap.TreeExplainer(fusion_estimator)
    shap_values = explainer.shap_values(x_train)
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{name}" for name in feature_names])
    shap_df.insert(0, "patient_id", train_ids)
    shap_df.insert(1, "mRS_binary", train_df["mRS_binary"].to_numpy())
    shap_df.insert(2, f"{model_name}_prob", train_prob)
    shap_df.to_csv(output_dir / f"shap_values_{model_name}.csv", index=False, encoding="utf-8-sig")

    shap.summary_plot(shap_values, x_train, show=False, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_shap_summary_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x_train, plot_type="bar", show=False, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_shap_bar_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    if model_name == "Fusion_stack_M1_M2_M3_M4":
        patient_idx = int(np.argmax(train_prob))
        shap.force_plot(
            explainer.expected_value,
            shap_values[patient_idx],
            x_train.iloc[patient_idx],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"fig_shap_force_{model_name}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_model_input_tables(selection: dict[str, FeatureSelectionResult], output_dir: Path) -> None:
    rows = []
    for model_name, result in selection.items():
        for order, feature in enumerate(result.selected_radiomics, start=1):
            rows.append({"Model": model_name, "Order": order, "Feature": feature})
        result.lasso_coef.to_csv(output_dir / f"lasso_coefficients_{model_name}.csv", index=False, encoding="utf-8-sig")
        result.lasso_cv.to_csv(output_dir / f"lasso_lambda_1se_curve_{model_name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": result.mrmr_features}).to_csv(output_dir / f"mrmr_features_{model_name}.csv", index=False, encoding="utf-8-sig")
        result.rfe_curve.to_csv(output_dir / f"rfe_curve_{model_name}.csv", index=False, encoding="utf-8-sig")
        result.rfe_ranking.to_csv(output_dir / f"rfe_ranking_{model_name}.csv", index=False, encoding="utf-8-sig")
        plot_rfe_curve(result.rfe_curve, f"RFE 1SE curve ({model_name})", output_dir / f"fig_rfe_curve_{model_name}.png")
    pd.DataFrame(rows).to_csv(output_dir / "selected_radiomics_features.csv", index=False, encoding="utf-8-sig")


def final_model_features_table(feature_sets: dict[str, list[str]], groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for model_name, features in feature_sets.items():
        for order, feature in enumerate(features, start=1):
            if feature in groups["clinical"]:
                feature_group = "clinical"
            elif feature in groups["pre"]:
                feature_group = "pre_radiomics"
            elif feature in groups["post"]:
                feature_group = "post_radiomics"
            elif feature in groups["delta"]:
                feature_group = "delta"
            else:
                feature_group = "other"
            rows.append(
                {
                    "Model": model_name,
                    "Order": order,
                    "Feature": feature,
                    "Feature_group": feature_group,
                }
            )
    return pd.DataFrame(rows)


def load_feature_sets_from_final_table(feature_table: Path) -> dict[str, list[str]]:
    """Load fixed model features from table_final_model_features.csv."""
    if not feature_table.exists():
        raise FileNotFoundError(f"Final feature table not found for test-only mode: {feature_table}")

    df = pd.read_csv(feature_table, encoding="utf-8-sig")
    required = {"Model", "Order", "Feature"}
    if not required.issubset(df.columns):
        raise ValueError(f"{feature_table} must contain columns: {sorted(required)}")

    feature_sets = {}
    for model_name in BASE_MODEL_NAMES:
        features = (
            df[df["Model"] == model_name]
            .sort_values("Order")["Feature"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not features:
            raise ValueError(f"No features found for {model_name} in the final feature table")
        feature_sets[model_name] = features

    return feature_sets


def validate_feature_sets_exist(datasets: dict[str, pd.DataFrame], feature_sets: dict[str, list[str]]) -> None:
    """Check that fixed features are present in every dataset."""
    for dataset_name, df in datasets.items():
        missing_by_model = {}
        columns = set(df.columns)
        for model_name, features in feature_sets.items():
            missing = [feature for feature in features if feature not in columns]
            if missing:
                missing_by_model[model_name] = missing
        if missing_by_model:
            details = "; ".join(
                f"{model}: {missing[:8]}{'...' if len(missing) > 8 else ''}"
                for model, missing in missing_by_model.items()
            )
            raise ValueError(f"{dataset_name} is missing required test-only features: {details}")


def test_only_main(
    dataset_specs: list[DatasetSpec],
    final_feature_table: Path,
    output_dir: Path,
    trained_model_dir: Path,
) -> None:
    """Run external testing with fixed features and saved base model bundles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _, datasets = load_datasets(dataset_specs)
    feature_sets = load_feature_sets_from_final_table(final_feature_table)
    validate_feature_sets_exist(datasets, feature_sets)
    bundles = load_trained_model_bundles(trained_model_dir)

    perf, thresholds, probabilities, fusion_models = predict_with_trained_model_bundles(datasets, bundles)
    train_df, _ = load_datasets(dataset_specs)
    perf_ci = compute_performance_ci_table(perf, probabilities, datasets, train_df=train_df)
    delong_table = compute_test_center_delong_table(probabilities, datasets)
    hl_table = compute_hl_calibration_table(probabilities, datasets)

    perf.to_csv(output_dir / "table_model_performance_test_only.csv", index=False, encoding="utf-8-sig")
    perf_ci.to_csv(output_dir / "table_model_performance_ci_test_only.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(output_dir / "table_youden_thresholds_and_confusion_test_only.csv", index=False, encoding="utf-8-sig")
    delong_table.to_csv(output_dir / "table_test_center_delong_test_only.csv", index=False, encoding="utf-8-sig")
    hl_table.to_csv(output_dir / "table_hl_calibration_test_only.csv", index=False, encoding="utf-8-sig")
    save_prediction_probabilities(datasets, probabilities, output_dir / "test_only_prediction_probabilities.csv")
    plot_dca_and_calibration(probabilities, datasets, output_dir)
    explained_fusions = best_fusion_model_names(perf)
    pd.DataFrame({"Model": explained_fusions}).to_csv(
        output_dir / "table_explained_fusion_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    explain_full_stacking_fusion(train_df, datasets, probabilities, fusion_models, output_dir, explained_fusions)

    print("Test-only mode complete")
    print(f"Fixed feature table: {final_feature_table}")
    print(f"Loaded model directory: {trained_model_dir}")
    print(f"Output directory: {output_dir}")
    print(perf.sort_values(["Dataset", "Model"])[["Dataset", "Model", "AUC", "ACC", "Sensitivity", "Specificity", "F1", "P_vs_M1"]])


def main(
    dataset_specs: list[DatasetSpec],
    clinical_cols: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, datasets = load_datasets(dataset_specs)
    all_cols = numeric_feature_columns(train_df)
    groups = classify_feature_groups(all_cols, clinical_cols)
    y_train = train_df["mRS_binary"].to_numpy()
    clinical_selection = select_clinical_features(train_df, groups["clinical"], y_train)

    feature_sets, selection = build_model_feature_sets(groups, train_df, y_train, clinical_selection.selected_clinical)
    perf, thresholds, fitted_models, probabilities, fusion_models = fit_and_predict(train_df, datasets, feature_sets)
    tuning_table = next(iter(fitted_models.values()))._analysis_tuning_table if fitted_models else pd.DataFrame()
    save_trained_model_bundles(output_dir, fitted_models, thresholds, fusion_models)
    perf_ci = compute_performance_ci_table(perf, probabilities, datasets, train_df=train_df)
    delong_table = compute_test_center_delong_table(probabilities, datasets)
    hl_table = compute_hl_calibration_table(probabilities, datasets)

    baseline = baseline_table(train_df, datasets, groups["clinical"])
    counts = feature_count_table(feature_sets, groups)
    final_features = final_model_features_table(feature_sets, groups)
    save_model_input_tables(selection, output_dir)

    baseline.to_csv(output_dir / "table_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    clinical_selection.univariate_table.to_csv(output_dir / "table_clinical_univariate_logistic.csv", index=False, encoding="utf-8-sig")
    clinical_selection.multivariate_table.to_csv(output_dir / "table_clinical_multivariate_logistic.csv", index=False, encoding="utf-8-sig")
    clinical_selection.rfe_curve.to_csv(output_dir / "clinical_rfe_curve.csv", index=False, encoding="utf-8-sig")
    clinical_selection.rfe_ranking.to_csv(output_dir / "clinical_rfe_ranking.csv", index=False, encoding="utf-8-sig")
    perf.to_csv(output_dir / "table_model_performance.csv", index=False, encoding="utf-8-sig")
    perf_ci.to_csv(output_dir / "table_model_performance_ci.csv", index=False, encoding="utf-8-sig")
    save_prediction_probabilities(datasets, probabilities, output_dir / "prediction_probabilities.csv")
    tuning_table.to_csv(output_dir / "table_xgboost_tuning.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(output_dir / "table_feature_counts.csv", index=False, encoding="utf-8-sig")
    final_features.to_csv(output_dir / "table_final_model_features.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(output_dir / "table_youden_thresholds_and_confusion.csv", index=False, encoding="utf-8-sig")
    delong_table.to_csv(output_dir / "table_test_center_delong.csv", index=False, encoding="utf-8-sig")
    hl_table.to_csv(output_dir / "table_hl_calibration.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / "mrs_xgboost_analysis_tables.xlsx") as writer:
        baseline.to_excel(writer, sheet_name="baseline", index=False)
        clinical_selection.univariate_table.to_excel(writer, sheet_name="clinical_univariate", index=False)
        clinical_selection.multivariate_table.to_excel(writer, sheet_name="clinical_multivariate", index=False)
        clinical_selection.rfe_curve.to_excel(writer, sheet_name="clinical_rfe_curve", index=False)
        clinical_selection.rfe_ranking.to_excel(writer, sheet_name="clinical_rfe_ranking", index=False)
        perf.to_excel(writer, sheet_name="performance", index=False)
        tuning_table.to_excel(writer, sheet_name="xgboost_tuning", index=False)
        counts.to_excel(writer, sheet_name="feature_counts", index=False)
        final_features.to_excel(writer, sheet_name="final_model_features", index=False)
        thresholds.to_excel(writer, sheet_name="youden_confusion", index=False)
        perf_ci.to_excel(writer, sheet_name="performance_ci", index=False)
        delong_table.to_excel(writer, sheet_name="test_center_delong", index=False)
        hl_table.to_excel(writer, sheet_name="hl_calibration", index=False)
        pd.DataFrame({"model": list(feature_sets), "features": [", ".join(feature_sets[m]) for m in feature_sets]}).to_excel(writer, sheet_name="model_features", index=False)

    plot_lasso(selection["M4"], train_df, y_train, output_dir, "M4")
    plot_rfe_curve(clinical_selection.rfe_curve, "Clinical RFE 1SE curve", output_dir / "fig_clinical_rfe_curve.png")
    plot_roc(probabilities, datasets, train_df, output_dir)
    plot_dca_and_calibration(probabilities, datasets, output_dir)
    plot_confusion_matrices(thresholds, output_dir)
    explain_all_models(train_df, datasets, perf, fitted_models, output_dir)
    explained_fusions = best_fusion_model_names(perf)
    pd.DataFrame({"Model": explained_fusions}).to_csv(
        output_dir / "table_explained_fusion_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    explain_full_stacking_fusion(train_df, datasets, probabilities, fusion_models, output_dir, explained_fusions)

    print(f"Training sample size: {len(train_df)}")
    print(f"Output directory: {output_dir}")
    print(f"Selected clinical variables: {clinical_selection.selected_clinical}")
    print(perf.sort_values(["Dataset", "Model"])[["Dataset", "Model", "AUC", "ACC", "Sensitivity", "Specificity", "F1", "P_vs_M1"]])


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "xgboost_mrs_results"
    TEST_ONLY_OUTPUT_DIR = OUTPUT_DIR / "test_only"
    FINAL_FEATURE_TABLE_FOR_TEST_ONLY = OUTPUT_DIR / "table_final_model_features.csv"
    TRAINED_MODEL_DIR = OUTPUT_DIR / "trained_models"

    # True uses fixed features and saved base models; False runs the full pipeline.
    ONLY_TEST_MODE = False

    DATASET_SPECS = [
        DatasetSpec("train_yjs", DATA_DIR / "featuresyjs.csv", "train"),
        DatasetSpec("train_tl", DATA_DIR / "featurestl.csv", "train"),
        DatasetSpec("train_fy", DATA_DIR / "featuresfy.csv", "train"),
        DatasetSpec("train_ay", DATA_DIR / "featuresay.csv", "train"),
        DatasetSpec("test_th", DATA_DIR / "featuresth.csv", "test"),
        DatasetSpec("test_efy", DATA_DIR / "featuresefy.csv", "test"),
    ]

    TEST_ONLY_DATASET_SPECS = [
        DatasetSpec("train_yjs", DATA_DIR / "featuresyjs.csv", "train"),
        DatasetSpec("train_tl", DATA_DIR / "featurestl.csv", "train"),
        DatasetSpec("train_fy", DATA_DIR / "featuresfy.csv", "train"),
        DatasetSpec("train_ay", DATA_DIR / "featuresay.csv", "train"),
        DatasetSpec("test_th", DATA_DIR / "featuresth_simulated_rank_test.csv", "test"),
        DatasetSpec("test_efy", DATA_DIR / "featuresefy_simulated_rank_test.csv", "test"),
    ]

    CLINICAL_COLS = [
        "Age",
        "Male",
        "mFS_score",
        "SEBES_score",
        "Acute_hydrocephalus",
        "GCS_score",
        "WFNS_score",
        "Hunt-Hess_score",
        "Posterior_circulation",
        "Size",
        "Hypertension",
        "Clipping",
    ]

    if ONLY_TEST_MODE or "--test-only" in sys.argv:
        test_only_main(TEST_ONLY_DATASET_SPECS, FINAL_FEATURE_TABLE_FOR_TEST_ONLY, TEST_ONLY_OUTPUT_DIR, TRAINED_MODEL_DIR)
    else:
        main(DATASET_SPECS, CLINICAL_COLS, OUTPUT_DIR)
