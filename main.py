import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "src" / "data" / "companies.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_model.pkl"


@st.cache_resource
def load_model():
    """Load the complete preprocessing and prediction pipeline once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 02_Model_Training.ipynb first."
        )
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_feature_data(feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Load the training data to build inputs with the same feature names."""
    data = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(feature_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing model features: {missing_columns}")
    return data[list(feature_columns)]


def build_prediction_input(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Collect one beginner-friendly input row from the Streamlit form."""
    values = {}
    with st.form("prediction_form"):
        st.subheader("Company details")
        for column in feature_data.columns:
            series = feature_data[column]
            if pd.api.types.is_numeric_dtype(series):
                default_value = float(series.median()) if series.notna().any() else 0.0
                values[column] = st.number_input(
                    label=column.replace("_", " ").title(),
                    value=default_value,
                    format="%.4f",
                )
            else:
                choices = sorted(series.dropna().astype(str).unique().tolist())
                if len(choices) <= 50 and choices:
                    values[column] = st.selectbox(
                        column.replace("_", " ").title(), choices
                    )
                else:
                    values[column] = st.text_input(
                        column.replace("_", " ").title(), value=""
                    )

        submitted = st.form_submit_button("Predict startup status")

    if submitted:
        return pd.DataFrame([values], columns=feature_data.columns)
    return pd.DataFrame()


st.set_page_config(page_title="Startup Status Classifier", page_icon="📊")
st.title("Startup Status Classifier")
st.write("Enter company information and predict its current startup status.")

try:
    model_artifact = load_model()
    model = model_artifact["pipeline"]
    target_encoder = model_artifact["target_encoder"]
    feature_columns = tuple(model_artifact["feature_columns"])
    feature_data = load_feature_data(feature_columns)
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

input_data = build_prediction_input(feature_data)
if not input_data.empty:
    prediction_code = model.predict(input_data).astype(int)
    prediction = target_encoder.inverse_transform(prediction_code)[0]
    st.success(f"Predicted startup status: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = target_encoder.inverse_transform(model.classes_.astype(int))
        probability_table = pd.DataFrame(
            {"Status": classes, "Probability": probabilities}
        ).sort_values("Probability", ascending=False)
        st.subheader("Prediction probabilities")
        st.dataframe(probability_table, hide_index=True, use_container_width=True)
