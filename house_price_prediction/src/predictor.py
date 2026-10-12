from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"


def load_model(path=MODEL_PATH):
    """Load the trained house price prediction pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def predict_price(model, input_data):
    """Predict the house price for the supplied property data."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame")

    if input_data.empty:
        raise ValueError("input_data must contain at least one row")

    predictions = model.predict(input_data)

    if len(predictions) == 0:
        raise ValueError("Model returned no predictions")

    return predictions

def explain_prediction(model, input_data, reference_data=None):
    """Explain a Linear Regression pipeline prediction relative to a typical property.

    Contributions are computed in the model's transformed feature space. The
    reference point is the mean transformed feature vector from the training
    dataset, so the returned contributions add up to the difference between
    the property's prediction and the model's prediction for a typical row.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame")
    if input_data.empty:
        raise ValueError("input_data must contain at least one row")
    if reference_data is None:
        raise ValueError("reference_data is required for a reliable explanation")
    if not isinstance(reference_data, pd.DataFrame) or reference_data.empty:
        raise ValueError("reference_data must be a non-empty pandas DataFrame")

    if not hasattr(model, "named_steps"):
        raise TypeError("Explanation requires a scikit-learn Pipeline")

    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None or not hasattr(estimator, "coef_"):
        raise TypeError("Explanation currently requires a Linear Regression pipeline")

    transformed_input = preprocessor.transform(input_data)
    transformed_reference = preprocessor.transform(reference_data)

    feature_names = preprocessor.get_feature_names_out()
    coefficients = estimator.coef_
    if getattr(coefficients, "ndim", 1) != 1:
        raise TypeError("Explanation currently supports single-output regression")

    contributions = (transformed_input[0] - transformed_reference.mean(axis=0)) * coefficients
    prediction = float(model.predict(input_data)[0])
    reference_prediction = float(
        estimator.intercept_ + (transformed_reference.mean(axis=0) * coefficients).sum()
    )

    friendly_names = {
        "numeric__area": "Area",
        "numeric__bedrooms": "Bedrooms",
        "numeric__bathrooms": "Bathrooms",
        "numeric__stories": "Stories",
        "numeric__parking": "Parking",
        "categorical__mainroad_yes": "Main road",
        "categorical__guestroom_yes": "Guest room",
        "categorical__basement_yes": "Basement",
        "categorical__hotwaterheating_yes": "Hot water heating",
        "categorical__airconditioning_yes": "Air conditioning",
        "categorical__prefarea_yes": "Preferred area",
        "categorical__furnishingstatus_semi-furnished": "Semi-furnished",
        "categorical__furnishingstatus_unfurnished": "Unfurnished",
    }

    rows = []
    for name, value in zip(feature_names, contributions):
        value = float(value)
        rows.append({
            "feature": friendly_names.get(name, name),
            "contribution": value,
            "raw_feature": name,
        })

    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "prediction": prediction,
        "reference_prediction": reference_prediction,
        "difference": prediction - reference_prediction,
        "contributions": rows,
    }

