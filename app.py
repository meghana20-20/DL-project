"""Streamlit front end for the V4 marine catch model (ONNX MLP)."""
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st

MODELS = Path(__file__).parent / "models"

# Columns used when the preprocessor was fitted (scripts/preprocess_v4.py)
CAT_COLS = ["FLEET", "FISHERY", "FISHERY_GROUP", "GEAR",
            "SPECIES", "SPECIES_CATEGORY", "FISHING_GROUND_CODE"]


@st.cache_resource
def load_artifacts():
    """Load the fitted preprocessor and the ONNX model once."""
    pre = joblib.load(MODELS / "v4_preprocessor.joblib")
    sess = ort.InferenceSession(
        str(MODELS / "v4_mlp_tuned.onnx"), providers=["CPUExecutionProvider"]
    )
    return pre, sess


def predict(pre, sess, row: dict) -> float:
    """Apply the same feature engineering as training, then predict catch (mt)."""
    df = pd.DataFrame([row])

    # Missingness flags, then fill effort with 0 (same as training)
    df["has_effort_hooks"] = df["total_effort_hooks"].notna().astype(int)
    df["has_effort_fdays"] = df["total_effort_fdays"].notna().astype(int)
    df["total_effort_hooks"] = df["total_effort_hooks"].fillna(0)
    df["total_effort_fdays"] = df["total_effort_fdays"].fillna(0)
    df["has_argo_temp"] = df["argo_temp"].notna().astype(int)
    df["has_argo_sal"] = df["argo_sal"].notna().astype(int)
    df["has_argo_match"] = df["nearest_argo_distance_km"].notna().astype(int)
    df["FISHING_GROUND_CODE"] = df["FISHING_GROUND_CODE"].astype(str)

    X = pre.transform(df)
    if hasattr(X, "toarray"):  # sparse -> dense
        X = X.toarray()
    X = X.astype(np.float32)

    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: X})[0]
    pred_log = float(np.ravel(out)[0])
    return pred_log, max(float(np.expm1(pred_log)), 0.0)


st.set_page_config(page_title="Marine Catch Predictor", page_icon="🐟")
st.title("🐟 Marine Catch Predictor")
st.write("Predict fishing catch (metric tonnes) from fleet, species, location "
         "and ocean conditions using the trained MLP.")

try:
    pre, sess = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files from the `models/` folder: {e}")
    st.stop()

categories = pre.named_transformers_["cat"].named_steps["ohe"].categories_
cat_options = {c: [str(v) for v in cats] for c, cats in zip(CAT_COLS, categories)}

st.subheader("Fishery details")
col1, col2 = st.columns(2)
row = {}
for i, c in enumerate(CAT_COLS):
    with (col1 if i % 2 == 0 else col2):
        row[c] = st.selectbox(c.replace("_", " ").title(), cat_options[c])

st.subheader("Location and time")
c1, c2, c3 = st.columns(3)
row["MONTH_START"] = c1.number_input("Month (1-12)", 1, 12, 6)
row["fishing_ground_lat"] = c2.number_input("Latitude", -90.0, 90.0, -10.0)
row["fishing_ground_lon"] = c3.number_input("Longitude", -180.0, 180.0, 60.0)
c4, c5 = st.columns(2)
row["grid_resolution"] = c4.number_input("Grid resolution", 0.0, 100.0, 5.0)
row["ocean_area_iotc_km2"] = c5.number_input("Ocean area (km²)", 0.0, 1e9, 0.0)

st.subheader("Fishing effort")
if st.checkbox("Effort data available", value=True):
    e1, e2, e3 = st.columns(3)
    row["total_effort_hooks"] = e1.number_input("Total effort (hooks)", 0.0, 1e9, 0.0)
    row["total_effort_fdays"] = e2.number_input("Total effort (fishing days)", 0.0, 1e9, 0.0)
    row["effort_record_count"] = e3.number_input("Effort record count", 0.0, 1e9, 1.0)
else:
    row["total_effort_hooks"] = np.nan
    row["total_effort_fdays"] = np.nan
    row["effort_record_count"] = np.nan

st.subheader("Ocean conditions (Argo)")
if st.checkbox("Argo float data available", value=True):
    a1, a2, a3 = st.columns(3)
    row["argo_temp"] = a1.number_input("Temperature (°C)", -5.0, 45.0, 25.0)
    row["argo_sal"] = a2.number_input("Salinity (PSU)", 0.0, 45.0, 35.0)
    row["nearest_argo_distance_km"] = a3.number_input("Distance to nearest float (km)", 0.0, 20000.0, 100.0)
else:
    row["argo_temp"] = np.nan
    row["argo_sal"] = np.nan
    row["nearest_argo_distance_km"] = np.nan

if st.button("Predict", type="primary"):
    try:
        pred_log, catch_mt = predict(pre, sess, row)
        st.success(f"Predicted catch: **{catch_mt:,.2f} metric tonnes**")
        st.caption(f"Raw model output (log1p scale): {pred_log:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("About this model"):
    st.write("MLP trained on the V4 marine catch dataset with a chronological "
             "train/validation/test split. Inputs are encoded with a scikit-learn "
             "preprocessor fitted on the training set only.")