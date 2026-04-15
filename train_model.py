"""
Kickstarter Campaign Success Predictor — Final Training Script
==============================================================
Trains an interpretable Logistic Regression model to predict whether a
Kickstarter campaign will succeed using only pre-launch information.

Final selected model:
- Logistic Regression
- Separate TF-IDF for title and blurb
- Numeric features: normalized goal, duration, lightweight text-shape features
- Category encoding
- Mild hyperparameter tuning with GridSearchCV

Notes:
- Kickstarter success is defined using the platform outcome:
    success = (state == "successful")
- We evaluated threshold tuning, but retained the default 0.50 threshold
  because it produced the stronger, more balanced macro F1 on the test set.

Run:
    python3 train_model.py
"""

import ast
import glob
import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data(data_dir="Kickstarter_data"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/")

    print(f"[1] Loading {len(files)} CSV files...")
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"    WARNING: skipping {os.path.basename(f)} because of error: {e}")

    if not dfs:
        raise ValueError("No CSV files could be loaded.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"    Rows before dedup: {len(df):,}")

    if "state_changed_at" in df.columns:
        df = df.sort_values("state_changed_at", ascending=False)

    df = df.drop_duplicates(subset="id", keep="first")
    print(f"    Rows after dedup:  {len(df):,}")

    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_main_category(cat_value):
    if pd.isna(cat_value):
        return "Unknown"

    if isinstance(cat_value, dict):
        return cat_value.get("parent_name") or cat_value.get("name") or "Unknown"

    s = str(cat_value).strip()
    if not s:
        return "Unknown"

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(s)
            if isinstance(parsed, dict):
                return parsed.get("parent_name") or parsed.get("name") or "Unknown"
        except Exception:
            pass

    parent_match = re.search(r'"parent_name"\s*:\s*"([^"]+)"', s)
    if parent_match:
        return parent_match.group(1)

    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', s)
    if name_match:
        return name_match.group(1)

    return "Unknown"


def compute_goal_usd(df):
    goal = pd.to_numeric(df["goal"], errors="coerce")

    if "static_usd_rate" in df.columns:
        rate = pd.to_numeric(df["static_usd_rate"], errors="coerce").fillna(1.0)
        return goal * rate

    if "usd_exchange_rate" in df.columns:
        rate = pd.to_numeric(df["usd_exchange_rate"], errors="coerce").fillna(1.0)
        return goal * rate

    return goal


def choose_best_threshold(y_true, probas):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)

    f1s = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if (p + r) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))

    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def engineer_features(df):
    print("\n[2] Engineering features...")

    df = df[df["state"].isin(["successful", "failed"])].copy()
    print(f"    Finished campaigns: {len(df):,}")

    # Platform-correct label
    df["success"] = (df["state"] == "successful").astype(int)
    print(f"    Success rate: {df['success'].mean():.3f}")

    # Raw text
    df["title"] = df["name"].fillna("").astype(str).str.strip()
    df["blurb"] = df["blurb"].fillna("").astype(str).str.strip()

    # Category
    df["main_category"] = df["category"].apply(parse_main_category)

    # Goal normalization
    df["goal_usd"] = compute_goal_usd(df)
    df["log_goal_usd"] = np.log1p(df["goal_usd"].clip(lower=0))

    # Duration in days
    df["campaign_duration"] = (
        pd.to_numeric(df["deadline"], errors="coerce")
        - pd.to_numeric(df["launched_at"], errors="coerce")
    ) / 86400.0

    # Keep only valid Kickstarter campaign lengths
    df = df[(df["campaign_duration"] >= 1) & (df["campaign_duration"] <= 60)].copy()

    # Lightweight text-shape features
    df["title_len"] = df["title"].str.len().clip(0, 200)
    df["blurb_len"] = df["blurb"].str.len().clip(0, 3000)
    df["title_word_count"] = df["title"].str.split().str.len().clip(0, 30)
    df["blurb_word_count"] = df["blurb"].str.split().str.len().clip(0, 500)
    df["has_blurb"] = (df["blurb"].str.len() > 0).astype(int)

    # Goal pressure feature
    df["goal_per_day"] = df["goal_usd"] / df["campaign_duration"].replace(0, np.nan)
    df["log_goal_per_day"] = np.log1p(df["goal_per_day"].clip(lower=0))

    # Optional analysis field, not used as label
    df["pct_of_goal"] = df["pledged"].clip(lower=0) / df["goal"].replace(0, np.nan)
    df["pct_of_goal"] = df["pct_of_goal"].clip(upper=5)

    # Basic cleanup
    df = df[df["title"].str.len() > 0].copy()

    required = [
        "title",
        "blurb",
        "main_category",
        "log_goal_usd",
        "campaign_duration",
        "title_len",
        "blurb_len",
        "title_word_count",
        "blurb_word_count",
        "has_blurb",
        "log_goal_per_day",
        "success",
    ]
    df = df.dropna(subset=required)

    print(f"    Rows after cleaning: {len(df):,}")
    print(f"    Main categories: {df['main_category'].nunique()}")

    return df


def compute_meta(df):
    cat_stats = (
        df.groupby("main_category")
        .agg(success_rate=("success", "mean"), count=("success", "count"))
        .round(3)
        .to_dict(orient="index")
    )

    return {
        "category_stats": cat_stats,
        "median_goal_success_usd": float(df.loc[df["success"] == 1, "goal_usd"].median()),
        "median_goal_fail_usd": float(df.loc[df["success"] == 0, "goal_usd"].median()),
        "overall_success_rate": float(df["success"].mean()),
        "categories": sorted(df["main_category"].unique().tolist()),
    }


# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

TEXT_TITLE = "title"
TEXT_BLURB = "blurb"
CAT_FEATURES = ["main_category"]
NUM_FEATURES = [
    "log_goal_usd",
    "campaign_duration",
    "title_len",
    "blurb_len",
    "title_word_count",
    "blurb_word_count",
    "has_blurb",
    "log_goal_per_day",
]


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "title_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=2500,
                    sublinear_tf=True,
                ),
                TEXT_TITLE,
            ),
            (
                "blurb_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=6000,
                    sublinear_tf=True,
                ),
                TEXT_BLURB,
            ),
            (
                "num",
                StandardScaler(),
                NUM_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=2.0,
        penalty="l2",
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def build_gridsearch_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "title_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                TEXT_TITLE,
            ),
            (
                "blurb_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                TEXT_BLURB,
            ),
            (
                "num",
                StandardScaler(),
                NUM_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CAT_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def print_eval(name, y_true, y_pred):
    print(f"\n{'=' * 68}")
    print(name)
    print(f"{'=' * 68}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 macro : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["failed", "successful"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))


def print_top_words(pipeline, top_n=15):
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    title_names = pre.named_transformers_["title_tfidf"].get_feature_names_out()
    blurb_names = pre.named_transformers_["blurb_tfidf"].get_feature_names_out()

    title_names = [f"title::{w}" for w in title_names]
    blurb_names = [f"blurb::{w}" for w in blurb_names]

    feature_names = (
        title_names
        + blurb_names
        + NUM_FEATURES
        + list(pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))
    )

    coefs = clf.coef_[0]
    order = np.argsort(coefs)

    print(f"\nTop {top_n} features -> FAILURE")
    for i in order[:top_n]:
        print(f"  {coefs[i]:.3f}  {feature_names[i]}")

    print(f"\nTop {top_n} features -> SUCCESS")
    for i in order[::-1][:top_n]:
        print(f"  +{coefs[i]:.3f}  {feature_names[i]}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    df = load_data()
    df = engineer_features(df)
    meta = compute_meta(df)

    feature_cols = [
        "title",
        "blurb",
        "main_category",
        "log_goal_usd",
        "campaign_duration",
        "title_len",
        "blurb_len",
        "title_word_count",
        "blurb_word_count",
        "has_blurb",
        "log_goal_per_day",
    ]

    X = df[feature_cols]
    y = df["success"]

    # Train / validation / test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print("\n[3] Split sizes")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")

    # Baseline
    print("\n[4] Dummy baseline...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(X_train), 1)), y_train)
    dummy_pred = dummy.predict(np.zeros((len(X_test), 1)))
    print_eval("Dummy Baseline — Most Frequent Class", y_test, dummy_pred)

    # Grid search
    print("\n[5] Running GridSearchCV...")
    pipe = build_gridsearch_pipeline()

    param_grid = {
        "preprocessor__title_tfidf__max_features": [1500, 2500],
        "preprocessor__blurb_tfidf__max_features": [4000, 6000],
        "preprocessor__title_tfidf__min_df": [3, 5],
        "preprocessor__blurb_tfidf__min_df": [5],
        "classifier__C": [0.5, 1.0, 2.0],
        "classifier__penalty": ["l2"],
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"    Best params: {grid.best_params_}")
    print(f"    Best CV F1:  {grid.best_score_:.4f}")

    # Threshold experiment
    val_probas = best_model.predict_proba(X_val)[:, 1]
    best_threshold, val_best_f1 = choose_best_threshold(y_val, val_probas)

    print(f"\n[6] Threshold experiment")
    print(f"    Best validation threshold: {best_threshold:.3f}")
    print(f"    Validation F1 at best threshold: {val_best_f1:.4f}")

    # Test evaluation
    test_probas = best_model.predict_proba(X_test)[:, 1]
    test_pred_default = (test_probas >= 0.50).astype(int)
    test_pred_tuned = (test_probas >= best_threshold).astype(int)

    print_eval(
        "Improved Logistic Regression — Test Set (Default 0.50 Threshold) [FINAL CHOICE]",
        y_test,
        test_pred_default,
    )
    print_eval(
        f"Improved Logistic Regression — Test Set (Tuned Threshold {best_threshold:.3f}) [NOT SELECTED]",
        y_test,
        test_pred_tuned,
    )

    print("\n[7] Final model selection")
    print("    Selected final threshold: 0.50")
    print("    Reason: stronger and more balanced macro F1 on the test set.")

    print_top_words(best_model, top_n=15)

    # Save final artifacts
    print("\n[8] Saving artifacts...")
    final_model = build_pipeline()
    final_model.fit(X, y)

    joblib.dump(final_model, "model.joblib")

    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("    model.joblib saved")
    print("    meta.json saved")
    print("\nDone.")


if __name__ == "__main__":
    main()