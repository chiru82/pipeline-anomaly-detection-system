from __future__ import annotations

import math
import random
import sys
import json
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents
sys.path.insert(0, str(ROOT))
import generate_pipeline_splits as simulator  # noqa: E402


FEATURES = [
    "flow_rate_l_min",
    "inlet_pressure_bar",
    "outlet_pressure_bar",
    "pipe_diameter_mm",
    "pipe_length_m",
    "fluid_type",
]
NUMERIC_FEATURES = FEATURES[:-1]
FLUID_CODES = {"Water": 0, "Oil": 1}
CLASS_NAMES = simulator.CLASS_NAMES
CLASS_IDS = list(CLASS_NAMES.keys())


def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame[FEATURES].copy()
    encoded["fluid_type"] = encoded["fluid_type"].astype(str).str.strip().str.title()
    if not encoded["fluid_type"].isin(FLUID_CODES).all():
        raise ValueError("Fluid type must be Water or Oil.")
    encoded = pd.get_dummies(encoded, columns=["fluid_type"], dtype=np.uint8)
    return encoded


@st.cache_resource(show_spinner=False)
def load_colab_model(root_text: str, app_text: str):
    root = Path(root_text)
    app_dir = Path(app_text)
    asset_dir = Path(tempfile.gettempdir()) / "pipeline_xgboost_outputs"
    asset_dir.mkdir(parents=True, exist_ok=True)
    packaged_files = [
        "pipeline_xgboost_model.json",
        "pipeline_model_metadata.json",
        "test_metrics.json",
        "simulated_normal_reference_ranges.csv",
        "test_confusion_matrix.png",
        "global_feature_importance.png",
        "test_precision_recall.png",
        "learning_curve.png",
    ]
    if not all((asset_dir / name).exists() for name in packaged_files):
        with zipfile.ZipFile(app_dir / "pipeline_xgboost_outputs.zip") as archive:
            for name in packaged_files:
                with archive.open(name) as source, (asset_dir / name).open("wb") as destination:
                    destination.write(source.read())

    metadata = json.loads((asset_dir / "pipeline_model_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((asset_dir / "test_metrics.json").read_text(encoding="utf-8"))
    model = XGBClassifier()
    model.load_model(str(asset_dir / "pipeline_xgboost_model.json"))
    train = pd.read_csv(root / "train_dataset_50000.csv")
    reference = pd.read_csv(asset_dir / "simulated_normal_reference_ranges.csv")

    class_names = {int(key): value for key, value in metadata["class_names"].items()}
    if set(class_names) != set(CLASS_IDS):
        raise ValueError("The saved model metadata does not contain the expected five classes.")
    feature_columns = metadata["feature_columns_after_one_hot"]
    importance = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    return model, metadata, metrics, importance, train, reference, asset_dir


@st.cache_data(show_spinner=False)
def make_demo_cases(root_text: str) -> pd.DataFrame:
    """Create ten fresh simulator cases (five classes for each fluid).

    The simulator uses a new seed and scenarios, then checks the exact six-input
    keys against the 65k combined file before exporting the demo CSV.
    """
    root = Path(root_text)
    combined = pd.read_csv(root / "combined_dataset_65000.csv")
    existing_keys = {
        simulator.feature_key(row)
        for row in combined[FEATURES].to_dict(orient="records")
    }
    rng = random.Random(20301112)
    new_rows: list[dict] = []
    next_index = 900_000

    for fluid_index, (fluid_name, fluid_properties) in enumerate(simulator.FLUIDS.items()):
        scenario_rows = None
        for attempt in range(500):
            candidate = simulator.create_scenario(
                rng,
                fluid_name,
                fluid_properties,
                "test",
                next_index + fluid_index * 1_000 + attempt,
            )
            if candidate is None:
                continue
            candidate_keys = [simulator.feature_key(row) for row in candidate]
            if any(key in existing_keys for key in candidate_keys):
                continue
            if len(candidate_keys) != len(set(candidate_keys)):
                continue
            scenario_rows = candidate
            existing_keys.update(candidate_keys)
            break

        if scenario_rows is None:
            raise RuntimeError(f"Could not create fresh demo cases for {fluid_name}.")

        for row in scenario_rows:
            new_rows.append(
                {
                    "demo_case_id": f"DEMO-{len(new_rows) + 1:02d}",
                    **{field: row[field] for field in FEATURES},
                    "simulator_reference_class_id": int(row["class_id"]),
                    "simulator_reference_class": row["class_label"],
                    "case_origin": "new simulated scenario; not a row in the 65k files",
                }
            )

    demo = pd.DataFrame(new_rows)
    if len(demo) != 10:
        raise RuntimeError(f"Expected 10 demo rows, created {len(demo)}.")
    demo_path = Path(__file__).resolve().parent / "demo_cases_10.csv"
    demo.to_csv(demo_path, index=False)
    return demo


def case_matches_input(case: pd.Series, values: dict) -> bool:
    if case["fluid_type"] != values["fluid_type"]:
        return False
    return all(
        math.isclose(float(case[field]), float(values[field]), rel_tol=0, abs_tol=1e-7)
        for field in NUMERIC_FEATURES
    )


st.set_page_config(
    page_title="Pipeline Anomaly Detection | Trial",
    page_icon="🟢",
    layout="wide",
)
st.markdown(
    """
    <style>
.stApp { background: linear-gradient(135deg, #eaf5ff 0%, #f7fbff 55%, #e3f2ff 100%); }
[data-testid="stMetric"] { background: white; border: 1px solid #c7def4;
      border-radius: 12px; padding: 14px 16px; }
div[data-testid="stSidebar"] { background: #e6f2ff; }
h1, h2, h3 { color: #083b78; }
.stApp h1 { font-size: 3rem; font-weight: 850; text-shadow: 0 2px 0 #b6d9ff; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; box-shadow: 0 8px 24px #0b4b8a18; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#087cf0,#0649ad);
  border: 0; box-shadow: 0 5px 16px #087cf055; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pipeline Anomaly Detection System")
st.caption("XGBoost pipeline analysis • leak and blockage classification")
st.info(
    "This dashboard uses your Colab-trained XGBoost model. Its evaluation data is simulated, "
    "so the test metrics are not proof of real-pipeline accuracy."
)

missing = [
    name
    for name, path in [
        ("train_dataset_50000.csv", ROOT / "train_dataset_50000.csv"),
        ("combined_dataset_65000.csv", ROOT / "combined_dataset_65000.csv"),
        ("generate_pipeline_splits.py", ROOT / "generate_pipeline_splits.py"),
        ("pipeline_xgboost_outputs.zip", Path(__file__).resolve().parent / "pipeline_xgboost_outputs.zip"),
    ]
    if not path.exists()
]
if missing:
    st.error("Required files are missing: " + ", ".join(missing))
    st.stop()

with st.spinner("Loading the XGBoost model trained in Colab…"):
    model, metadata, metrics, importance, train_data, reference_data, model_assets_dir = load_colab_model(
        str(ROOT), str(Path(__file__).resolve().parent)
    )
with st.spinner("Creating ten fresh demo cases and checking they are absent from the 65k inputs…"):
    demo_cases = make_demo_cases(str(ROOT))
CLASS_NAMES = {int(key): value for key, value in metadata["class_names"].items()}
temperature = float(metadata["calibration_temperature"])
feature_columns = metadata["feature_columns_after_one_hot"]


def calibrated_probabilities(model, encoded_input):
    raw = model.predict_proba(encoded_input)
    model_classes = list(getattr(model, "classes_", CLASS_IDS))
    ordered = raw[:, [model_classes.index(class_id) for class_id in CLASS_IDS]]
    logits = np.log(np.clip(ordered, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

st.sidebar.header("Input pipeline parameters")
sample_ids = demo_cases["demo_case_id"].tolist()
selected_demo_id = st.sidebar.selectbox("Optional sample case", ["Manual input"] + sample_ids)
if st.sidebar.button("Load selected sample") and selected_demo_id != "Manual input":
    selected = demo_cases.loc[demo_cases["demo_case_id"] == selected_demo_id].iloc[0]
    for field in NUMERIC_FEATURES:
        st.session_state[f"input_{field}"] = float(selected[field])
    st.session_state["input_fluid_type"] = selected["fluid_type"]
    st.session_state["loaded_demo_id"] = selected_demo_id
    st.rerun()

loaded_id = st.session_state.get("loaded_demo_id", "")
loaded_case = None
if loaded_id in sample_ids:
    loaded_case = demo_cases.loc[demo_cases["demo_case_id"] == loaded_id].iloc[0]

defaults = {
    "flow_rate_l_min": 75.0,
    "inlet_pressure_bar": 6.0,
    "outlet_pressure_bar": 5.8,
    "pipe_diameter_mm": 100.0,
    "pipe_length_m": 50.0,
}
for field, default in defaults.items():
    st.session_state.setdefault(f"input_{field}", default)
st.session_state.setdefault("input_fluid_type", "Water")

with st.sidebar.form("pipeline_input_form"):
    flow = st.number_input(
        "Flow rate (L/min)", min_value=0.0,
        format="%.6f", key="input_flow_rate_l_min",
    )
    inlet = st.number_input(
        "Inlet pressure P₁ (bar)", min_value=0.001,
        format="%.6f",
        key="input_inlet_pressure_bar",
    )
    outlet = st.number_input(
        "Outlet pressure P₂ (bar)", min_value=0.001,
        format="%.6f",
        key="input_outlet_pressure_bar",
    )
    diameter = st.number_input(
        "Pipe diameter (mm)", min_value=0.1,
        format="%.6f",
        key="input_pipe_diameter_mm",
    )
    length = st.number_input(
        "Pipe length (m)", min_value=0.1,
        format="%.6f",
        key="input_pipe_length_m",
    )
    fluid = st.selectbox(
        "Fluid type", ["Water", "Oil"], key="input_fluid_type",
    )
    analyze = st.form_submit_button("Analyze pipeline", type="primary", use_container_width=True)

values = {
    "flow_rate_l_min": float(flow),
    "inlet_pressure_bar": float(inlet),
    "outlet_pressure_bar": float(outlet),
    "pipe_diameter_mm": float(diameter),
    "pipe_length_m": float(length),
    "fluid_type": fluid,
}

if analyze:
    errors = []
    if values["inlet_pressure_bar"] <= values["outlet_pressure_bar"]:
        errors.append("For this demo, inlet pressure must be greater than outlet pressure.")
    if values["outlet_pressure_bar"] <= 0:
        errors.append("Outlet pressure must be positive.")
    if values["pipe_diameter_mm"] <= 0 or values["pipe_length_m"] <= 0:
        errors.append("Pipe diameter and length must be positive.")
    if values["flow_rate_l_min"] < 0:
        errors.append("Flow rate cannot be negative.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        input_frame = pd.DataFrame([values], columns=FEATURES)
        encoded_input = encode_features(input_frame).reindex(columns=feature_columns, fill_value=0)
        probabilities = calibrated_probabilities(model, encoded_input)[0]
        predicted_id = int(CLASS_IDS[np.argmax(probabilities)])
        score = float(probabilities[predicted_id])
        pressure_drop = values["inlet_pressure_bar"] - values["outlet_pressure_bar"]
        diameter_m = values["pipe_diameter_mm"] / 1000.0
        area_m2 = math.pi * diameter_m**2 / 4.0
        flow_m3_s = values["flow_rate_l_min"] / 60_000.0
        velocity = flow_m3_s / area_m2

        loaded_match = loaded_case is not None and case_matches_input(loaded_case, values)
        card_a, card_b, card_c, card_d = st.columns(4)
        card_a.metric("Predicted class", CLASS_NAMES[predicted_id])
        card_b.metric("Top model score", f"{score * 100:.1f}%")
        card_c.metric("Pressure drop", f"{pressure_drop:.3f} bar")
        card_d.metric("Derived velocity", f"{velocity:.3f} m/s")
        st.caption(
            "The top model score is not the model's test accuracy and is not a calibrated real-world confidence."
        )

        if loaded_match:
            reference_label = loaded_case["simulator_reference_class"]
            st.info(
                f"Fresh demo case {loaded_case['demo_case_id']} — simulator reference label: "
                f"**{reference_label}**. This is a simulated reference, not a measured field diagnosis."
            )

        current_range = train_data[train_data["fluid_type"] == fluid]
        outside = []
        for field in NUMERIC_FEATURES:
            low = float(current_range[field].min())
            high = float(current_range[field].max())
            if not low <= values[field] <= high:
                outside.append(f"{field} ({low:.3g}–{high:.3g} in training data)")
        if outside:
            st.warning("Outside the training range for this fluid: " + ", ".join(outside))

        result_left, result_right = st.columns([1, 1])
        with result_left:
            st.subheader("Class scores")
            score_frame = pd.DataFrame(
                {"Estimated model score": probabilities},
                index=[CLASS_NAMES[i] for i in CLASS_IDS],
            )
            st.bar_chart(score_frame)
        with result_right:
            st.subheader("Pressure endpoints")
            st.caption("Illustrative straight-line interpolation; only inlet and outlet values were entered.")
            distances = np.linspace(0.0, values["pipe_length_m"], 21)
            pressure_values = np.linspace(values["inlet_pressure_bar"], values["outlet_pressure_bar"], 21)
            pressure_frame = pd.DataFrame(
                {"Illustrative pressure (bar)": pressure_values}, index=distances
            )
            pressure_frame.index.name = "Distance along pipe (m)"
            st.line_chart(pressure_frame)

        st.subheader(f"Input compared with simulated normal cases ({fluid})")
        fluid_reference = reference_data[reference_data["fluid_type"] == fluid].copy()
        fluid_reference["Your input"] = fluid_reference["parameter"].map(values)
        st.dataframe(fluid_reference, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Model performance on the held-out 5k test split")
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
metric_b.metric("Macro precision", f"{metrics['macro_precision'] * 100:.2f}%")
metric_c.metric("Macro recall", f"{metrics['macro_recall'] * 100:.2f}%")
metric_d.metric("Macro F1", f"{metrics['macro_f1'] * 100:.2f}%")
st.caption("These are aggregate simulated-test metrics, not the accuracy for one website input.")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("**Global feature importance**")
    st.bar_chart(importance.set_index("Feature"))
    st.caption("Importance is model-wide; it does not explain an individual prediction.")
with chart_right:
    st.markdown("**Test confusion matrix**")
    st.image(str(model_assets_dir / "test_confusion_matrix.png"), use_container_width=True)

with st.expander("About the ten demo cases"):
    st.write(
        "They are generated from new simulator scenarios with a separate seed, one case for each "
        "of the five classes for Water and Oil. The app checks their exact six-input combinations "
        "against the combined 65k file and writes them to `website_trial/demo_cases_10.csv`. "
        "They are useful for trying the interface, but they are still generated by the same "
        "simulator and are not independent real-world validation data."
    )
    st.download_button(
        "Download the 10 demo cases",
        data=demo_cases.to_csv(index=False).encode("utf-8"),
        file_name="demo_cases_10.csv",
        mime="text/csv",
    )

