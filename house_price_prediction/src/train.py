import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "house_price_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

NUMERIC_FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

REQUIRED_COLUMNS = set(
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["price"]
)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if (df["price"] <= 0).any():
        raise ValueError("Price values must be greater than zero.")

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            )

    df = df.drop_duplicates()

    if df.empty:
        raise ValueError("Dataset contains no rows after removing duplicates.")

    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def main():
    df = load_data()

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    results = {}

    print("Model comparison using 5-fold cross-validation")
    print("=" * 55)

    for name, model in models.items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
                "R2": "r2",
            },
        )

        mae = -scores["test_MAE"].mean()
        rmse = -scores["test_RMSE"].mean()
        r2 = scores["test_R2"].mean()

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

        print(f"\n{name}")
        print(f"CV MAE:  {mae:,.2f}")
        print(f"CV RMSE: {rmse:,.2f}")
        print(f"CV R2:   {r2:.4f}")

    best_model_name = max(
        results,
        key=lambda name: results[name]["R2"],
    )

    print("\n" + "=" * 55)
    print(f"Best model: {best_model_name}")

    best_pipeline = build_pipeline(models[best_model_name])
    best_pipeline.fit(X_train, y_train)

    test_predictions = best_pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = mean_squared_error(
        y_test,
        test_predictions,
    ) ** 0.5
    test_r2 = r2_score(y_test, test_predictions)

    print("\nFinal test-set performance")
    print("=" * 55)
    print(f"MAE:  {test_mae:,.2f}")
    print(f"RMSE: {test_rmse:,.2f}")
    print(f"R2:   {test_r2:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)

    metrics = {
        "selected_model": best_model_name,
        "cross_validation": results,
        "test_set": {
            "MAE": test_mae,
            "RMSE": test_rmse,
            "R2": test_r2,
        },
    }

    with METRICS_PATH.open("w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nPipeline saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
