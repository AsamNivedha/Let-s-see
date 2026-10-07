# House Price Predictor

House Price Predictor learning web application that estimates house prices based on property characteristics. It combines a trained machine learning pipeline with an interactive Streamlit interface to provide quick price estimates from user-provided property details.

## Features

- Predict house prices using a trained machine learning model
- Interactive and user-friendly Streamlit interface
- Supports numerical and categorical property features
- Displays the estimated price along with the selected property details
- Automated model testing using Pytest
- Separate training, prediction, and application components

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core development |
| Pandas | Data processing |
| Scikit-learn | Machine learning |
| Streamlit | Web application |
| Joblib | Model serialization |
| Pytest | Automated testing |

## How It Works

```text
Property Details
       ↓
Input Processing
       ↓
Trained ML Pipeline
       ↓
Price Prediction
       ↓
Estimated House Price
```

The application takes property characteristics such as area, bedrooms, bathrooms, stories, parking, furnishing status, and other property features. These inputs are processed and passed to the trained machine learning pipeline to generate an estimated house price.

## Project Structure

```text
AI-ML/
├── .github/
│   └── workflows/
│       └── tests.yml
├── models/
│   └── house_price_pipeline.pkl
├── tests/
│   └── test_model.py
├── src/
│   ├── predict.py
│   ├── predictor.py
│   ├── train.py
│   └── train.ipynb
├── data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application with:

```bash
streamlit run src/predict.py
```

The application will launch in your browser and allow you to enter property details and generate a prediction.

## Run Tests

The project includes automated tests for the machine learning components.

```bash
pytest -v
```

## Machine Learning

The trained model is stored as:

```text
models/house_price_pipeline.pkl
```

The project separates model training, prediction logic, and the user interface to keep the application modular and easier to maintain.

## Disclaimer

HomeValue provides machine learning-based estimates using the available training data and the characteristics entered by the user. The predictions are estimates and should not be considered guaranteed market prices or professional property valuations.
