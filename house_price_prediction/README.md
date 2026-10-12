
# House Price Predictor

A machine learning web application that estimates house prices based on property characteristics using a trained Scikit-learn pipeline and an interactive Streamlit interface.

## Features

- Predict house prices using user-provided property details.
- Interactive Streamlit interface for entering property characteristics.
- Integrated machine learning pipeline for price estimation.
- Prediction explanations and property insights.
- Automated testing using Pytest.
- Continuous integration using GitHub Actions.

## Tech Stack

Python · Pandas · Scikit-learn · Streamlit · Joblib · Pytest

## How It Works

Property Details → Input Processing → Trained ML Pipeline → Price Prediction → Estimated House Price

## Issues Identified & Fixes Implemented

- **Application launch:** Running `streamlit run src/predict.py` caused `ModuleNotFoundError: No module named 'src'`; fixed by launching the application from the project root using `python -m streamlit run src/predict.py`.
- **Prediction integration:** Integrated the trained `house_price_pipeline.pkl` model into the prediction workflow to generate estimated prices from user-entered property details.
- **Code structure:** Separated the application interface, prediction logic, and model training into `predict.py`, `predictor.py`, and `train.py` for improved maintainability.
- **Automated testing:** Implemented Pytest coverage for the model and prediction workflow and verified that all 16 tests passed.
- **Continuous integration:** Configured `.github/workflows/tests.yml` to install dependencies and execute automated tests on pushes and pull requests.
- **Documentation:** Updated the README with the project structure, setup instructions, application launch command, testing steps, and prediction disclaimer.

## Project Structure

```text
house_price_prediction/
├── .github/
│   └── workflows/
│       └── tests.yml
├── models/
│   ├── house_price_pipeline.pkl
│   └── model_metrics.json
├── src/
│   ├── predict.py
│   ├── predictor.py
│   ├── train.py
│   └── train.ipynb
├── tests/
│   └── test_model.py
├── data.csv
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

From the `house_price_prediction` project directory, run:

```bash
python -m streamlit run src/predict.py
```

## Run Tests

Execute the automated test suite:

```bash
python -m pytest -v
```

## Testing and Continuous Integration

The project includes automated tests for model loading, prediction logic, explanation consistency, and error handling. GitHub Actions runs the test suite on pushes and pull requests.

## Disclaimer

Predictions are estimates based on the available training data and user-provided property details. They are not guaranteed market prices or professional property valuations.
