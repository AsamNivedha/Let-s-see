from pathlib import Path

import pytest

import joblib
import pandas as pd

from src.train import build_pipeline, load_data
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
DATA_PATH = BASE_DIR / "data.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "parking": [2],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
    })

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] > 0


def test_dataset_has_required_columns():
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "price",
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus",
    }

    assert required_columns.issubset(df.columns)

def test_training_pipeline_can_be_created():
    pipeline = build_pipeline(LinearRegression())

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_training_pipeline_can_fit_and_predict():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("price", axis=1).head(20)
    y = df["price"].head(20)

    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == 20
    assert all(predictions > 0)

def test_load_data_rejects_missing_columns(tmp_path):
    invalid_data = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "price": [5000000],
    })

    file_path = tmp_path / "invalid.csv"
    invalid_data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(file_path)


def test_load_data_rejects_empty_dataset(tmp_path):
    empty_data = pd.DataFrame(columns=[
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus",
        "price",
    ])

    file_path = tmp_path / "empty.csv"
    empty_data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_data(file_path)


def test_load_data_rejects_missing_values(tmp_path):
    invalid_data = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "parking": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
        "price": [None],
    })

    file_path = tmp_path / "invalid.csv"
    invalid_data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_data(file_path)


def test_load_data_rejects_invalid_price(tmp_path):
    invalid_data = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "parking": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
        "price": [-100],
    })

    file_path = tmp_path / "invalid.csv"
    invalid_data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Price values must be greater than zero"):
        load_data(file_path)


def test_load_model():
    from src.predictor import load_model

    model = load_model()
    assert model is not None


def test_predict_price_returns_prediction():
    from src.predictor import load_model, predict_price

    model = load_model()

    sample = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "parking": [2],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
    })

    predictions = predict_price(model, sample)

    assert len(predictions) == 1
    assert predictions[0] > 0


def test_predict_price_rejects_empty_input():
    from src.predictor import load_model, predict_price

    model = load_model()
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError, match="at least one row"):
        predict_price(model, empty_data)


def test_predict_price_rejects_non_dataframe_input():
    from src.predictor import load_model, predict_price

    model = load_model()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict_price(model, {"area": 5000})


def test_load_model_rejects_missing_file(tmp_path):
    from src.predictor import load_model

    missing_model = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(missing_model)


def test_explain_prediction_is_consistent():
    from src.predictor import explain_prediction, load_model

    model = load_model()
    sample = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "parking": [2],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
    })
    reference = pd.read_csv(DATA_PATH).drop(columns=["price"])

    explanation = explain_prediction(model, sample, reference)
    contribution_sum = sum(item["contribution"] for item in explanation["contributions"])

    assert explanation["prediction"] > 0
    assert explanation["reference_prediction"] > 0
    assert contribution_sum == pytest.approx(explanation["difference"], rel=1e-8, abs=1e-4)
    assert explanation["prediction"] == pytest.approx(
        explanation["reference_prediction"] + contribution_sum,
        rel=1e-8,
        abs=1e-4,
    )


def test_explain_prediction_requires_reference_data():
    from src.predictor import explain_prediction, load_model

    model = load_model()
    sample = pd.DataFrame({"area": [5000]})

    with pytest.raises(ValueError, match="reference_data is required"):
        explain_prediction(model, sample)
