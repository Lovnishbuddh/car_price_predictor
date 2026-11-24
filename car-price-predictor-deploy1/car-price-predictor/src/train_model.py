"""
train_model.py
==============
Trains and evaluates the Car Price Predictor model.

What it does (the complete ML workflow):
    1. Load and preprocess the data (uses src/preprocess.py)
    2. Split into training (80%) and testing (20%) sets
    3. Build a scikit-learn Pipeline that:
         - one-hot encodes categorical columns (name, company, fuel_type)
         - feeds everything to the regression model
    4. Train THREE models and compare them fairly:
         - Linear Regression        (simple, interpretable baseline)
         - Random Forest Regressor  (ensemble of decision trees)
         - Extra Trees Regressor    (Random Forest's close cousin)
    5. Evaluate on the test set with R2, MAE, MSE, RMSE
       + 5-fold cross-validation for reliability
    6. Save the BEST model with joblib, together with a
       metadata JSON (metrics, dropdown options) that the
       Flask app reads.

Run:  python -m src.train_model
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.preprocess import (prepare_dataset, NUMERIC_FEATURES,
                            CATEGORICAL_FEATURES, TARGET)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def build_pipeline(model):
    """
    Create a scikit-learn Pipeline.

    ColumnTransformer applies DIFFERENT preprocessing to DIFFERENT columns:
        - numeric columns      -> passed through unchanged
        - categorical columns  -> OneHotEncoder creates 0/1 columns
          (e.g. fuel_type='Diesel' becomes fuel_type_Diesel=1, others=0)

    handle_unknown='ignore' lets the model still predict for a model name
    it has never seen during training (its columns just become 0).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test, X, y, cv):
    """Compute test-set metrics + cross-validation score for one model."""
    pred = pipeline.predict(X_test)
    metrics = {
        "model_name": name,
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 0),
        "mse": round(float(mean_squared_error(y_test, pred)), 0),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 0),
    }
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)
    return metrics, pred


def plot_feature_importance(pipeline, out_path):
    """Horizontal bar chart of which features matter most for the model."""
    if not hasattr(pipeline.named_steps["model"], "feature_importances_"):
        return  # Linear Regression has no feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [n.split("__", 1)[-1] for n in feature_names]
    importances = pipeline.named_steps["model"].feature_importances_

    # Group the one-hot columns back into their original feature
    def group(n):
        if n.startswith("fuel_type_"):
            return "fuel_type"
        if n.startswith("company_"):
            return "company (brand)"
        if n.startswith("name_"):
            return "name (model)"
        return n

    imp = (pd.Series(importances, index=feature_names)
           .groupby(group).sum().sort_values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_title("Which features matter most? (model importance)")
    ax.set_xlabel("Importance (share of prediction decisions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------ 1. load + preprocess
    df = prepare_dataset()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    print(f"Dataset ready: {len(df)} rows, {X.shape[1]} features")
    print(f"Target: {TARGET} (Rupees)  |  mean = Rs {y.mean():,.0f}, "
          f"min = Rs {y.min():,.0f}, max = Rs {y.max():,.0f}")

    # ------------------------------------------------ 2. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Training rows: {len(X_train)}   Test rows: {len(X_test)}")

    # ------------------------------------------------ 3. train candidates
    candidates = {
        "Linear Regression": build_pipeline(LinearRegression()),
        "Random Forest": build_pipeline(RandomForestRegressor(
            n_estimators=300, random_state=42)),
        "Extra Trees": build_pipeline(ExtraTreesRegressor(
            n_estimators=300, random_state=42)),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    all_metrics = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        m, _ = evaluate(name, pipe, X_test, y_test, X, y, cv)
        all_metrics[name] = m
        print(f"\n--- {name} ---")
        print(f"  R2   : {m['r2']}   (higher is better, max 1.0)")
        print(f"  MAE  : Rs {m['mae']:,.0f}  (average error)")
        print(f"  MSE  : {m['mse']:,.0f}")
        print(f"  RMSE : Rs {m['rmse']:,.0f}")
        print(f"  5-fold CV R2: {m['cv_r2_mean']} +/- {m['cv_r2_std']}")

    # ------------------------------------------------ 4. pick the winner
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["cv_r2_mean"])
    best_pipe = candidates[best_name]
    print(f"\n>>> Best model (by cross-validation): {best_name} "
          f"(CV R2 = {all_metrics[best_name]['cv_r2_mean']})")

    # ------------------------------------------------ 5. save artefacts
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    plot_feature_importance(best_pipe,
                            os.path.join("static", "images", "eda",
                                         "09_feature_importance.png"))

    # Dropdown options + validation ranges for the Flask app (single
    # source of truth - the web form is generated from the real data)
    metadata = {
        "best_model": best_name,
        "metrics": all_metrics,
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "car_models": sorted(df["name"].unique().tolist()),
        "companies": sorted(df["company"].unique().tolist()),
        "fuel_types": sorted(df["fuel_type"].unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "price_unit": "rupees",
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {META_PATH}")

    # ------------------------------------------------ 6. sample predictions
    print("\nSanity check - sample predictions on test data:")
    sample = X_test.head(5)
    sample_pred = best_pipe.predict(sample)
    for i, (_, row) in enumerate(sample.iterrows()):
        actual = y_test.iloc[i]
        print(f"  {row['name']:<28s} predicted Rs {sample_pred[i]:>12,.0f} "
              f"| actual Rs {actual:>12,.0f}")


if __name__ == "__main__":
    main()
