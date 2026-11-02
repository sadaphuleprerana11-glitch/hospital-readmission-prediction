import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==================================================
# MODERN UI DESIGN
# ==================================================
st.markdown("""
<style>
:root {
    --navy: #10263f;
    --teal: #087f8c;
    --ink: #19344f;
}
.stApp {
    background: linear-gradient(135deg, #f4f8fc, #eaf2f7);
    color: var(--ink);
}
.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10263f, #10475a);
}
[data-testid="stSidebar"] * {
    color: white;
}
h1, h2, h3 {
    color: #19344f;
    font-weight: 750;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce7ef;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(18,47,72,.07);
    min-height: 118px;
}
[data-testid="stMetricValue"] {
    color: #183a59;
    overflow-wrap: anywhere;
    white-space: normal;
}
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: white;
    border: 1px solid #dce7ef;
    border-radius: 16px;
    padding: 16px;
}
.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background: #087f8c;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 650;
}
.stButton button:hover,
.stFormSubmitButton button:hover,
.stDownloadButton button:hover {
    background: #066673;
    color: white;
}
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}
.hero {
    background: linear-gradient(115deg, #10263f, #087f8c);
    color: white;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 18px;
}
.hero h1 {
    color: white !important;
}
.hero p {
    color: #e7f5f8;
}
hr {
    border-color: #d7e3ec;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# FILE PATHS
# Keep app.py, model file and results CSV in one folder.
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hospital_readmission_model.joblib"
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"

# ==================================================
# HELPER FUNCTIONS
# ==================================================
@st.cache_resource
def load_model(path):
    """Load the trained machine-learning pipeline once."""
    return joblib.load(path)


def get_expected_columns(model):
    """
    Read the feature names expected by the fitted preprocessing step.
    This app expects a Pipeline containing a step named 'preprocessor'.
    """
    if not hasattr(model, "named_steps"):
        raise ValueError(
            "The saved model is not a scikit-learn Pipeline. "
            "Please check the model file used during training."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "The pipeline does not contain a step named 'preprocessor'. "
            "Update this step name to match the training pipeline."
        )

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "The fitted preprocessor does not expose feature_names_in_. "
            "Check that the model was trained using a pandas DataFrame."
        )

    return list(preprocessor.feature_names_in_)


def save_prediction_history(history_row):
    """Append one prediction to the CSV history file."""
    new_record = pd.DataFrame([history_row])

    if HISTORY_PATH.exists():
        old_history = pd.read_csv(HISTORY_PATH)
        updated_history = pd.concat(
            [old_history, new_record],
            ignore_index=True
        )
    else:
        updated_history = new_record

    updated_history.to_csv(HISTORY_PATH, index=False)


# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <h1>🏥 Hospital Readmission Prediction</h1>
    <p>AI Healthcare Analytics | MCA Data Science Research Project</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype only. This model and its experimental "
    "13% threshold are not clinically validated. "
    "Do not use predictions for patient-care decisions."
)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
page = st.sidebar.radio(
    "Navigation",
    ["Make Prediction", "Dashboard"],
    index=0
)
st.sidebar.divider()
st.sidebar.caption("Hospital Readmission Research Project")

# ==================================================
# PAGE 1: MAKE PREDICTION
# ==================================================
if page == "Make Prediction":

    st.header("🏥 Make a Prediction")
    st.write("Enter the patient encounter details below.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(
            "The model could not be loaded. This can happen when the "
            "scikit-learn version used to save the model differs from "
            f"the installed version. Details: {exc}"
        )
        st.stop()

    medication_options = [
        "metformin", "repaglinide", "nateglinide",
        "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]

    with st.form("patient_form"):

        st.subheader("Patient Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            race = st.selectbox(
                "Race",
                [
                    "Caucasian", "AfricanAmerican",
                    "Asian", "Hispanic", "Other", "Unknown"
                ]
            )

        with c2:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Unknown"]
            )

        with c3:
            age = st.selectbox(
                "Age Group",
                [
                    "[0-10)", "[10-20)", "[20-30)",
                    "[30-40)", "[40-50)", "[50-60)",
                    "[60-70)", "[70-80)", "[80-90)",
                    "[90-100)"
                ]
            )

        st.subheader("Admission Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            admission_type_id = st.number_input(
                "Admission Type ID", 1, 8, 1
            )

        with c2:
            discharge_disposition_id = st.number_input(
                "Discharge Disposition ID", 1, 30, 1
            )

        with c3:
            admission_source_id = st.number_input(
                "Admission Source ID", 1, 25, 1
            )

        st.subheader("Hospital Stay and Previous Visits")
        c1, c2, c3 = st.columns(3)

        with c1:
            time_in_hospital = st.number_input(
                "Time in Hospital (days)", 1, 30, 3
            )
            num_lab_procedures = st.number_input(
                "Number of Lab Procedures", 0, 150, 40
            )
            num_procedures = st.number_input(
                "Number of Procedures", 0, 10, 1
            )
            num_medications = st.number_input(
                "Number of Medications", 0, 100, 10
            )

        with c2:
            number_outpatient = st.number_input(
                "Previous Outpatient Visits", min_value=0, value=0
            )
            number_emergency = st.number_input(
                "Previous Emergency Visits", min_value=0, value=0
            )
            number_inpatient = st.number_input(
                "Previous Inpatient Visits", min_value=0, value=0
            )
            number_diagnoses = st.number_input(
                "Number of Diagnoses", 1, 20, 5
            )

        with c3:
            diag_1 = st.text_input("Primary Diagnosis Code", "250.8")
            diag_2 = st.text_input("Secondary Diagnosis Code", "276")
            diag_3 = st.text_input("Additional Diagnosis Code", "250")

        st.subheader("Medication Information")
        medication_values = {}
        med_cols = st.columns(3)

        for i, med in enumerate(medication_options):
            with med_cols[i % 3]:
                medication_values[med] = st.selectbox(
                    med,
                    ["No", "Steady", "Up", "Down"],
                    index=0,
                    key=f"med_{med}"
                )

        c1, c2 = st.columns(2)

        with c1:
            change = st.selectbox(
                "Medication Change",
                ["No", "Ch"]
            )

        with c2:
            diabetes_med = st.selectbox(
                "Diabetes Medication",
                ["Yes", "No"]
            )

        submitted = st.form_submit_button(
            "Predict Readmission Risk",
            use_container_width=True
        )

    # ==================================================
    # PREDICTION PROCESS
    # ==================================================
    if submitted:

        row = {
            "race": race,
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "change": change,
            "diabetesMed": diabetes_med,
            **medication_values
        }

        input_df = pd.DataFrame([row])

        try:
            expected_columns = get_expected_columns(model)

            # Check the input schema against the trained model schema.
            missing_columns = [
                col for col in expected_columns
                if col not in input_df.columns
            ]
            extra_columns = [
                col for col in input_df.columns
                if col not in expected_columns
            ]

            if missing_columns or extra_columns:
                raise ValueError(
                    f"Input feature mismatch. Missing: {missing_columns}; "
                    f"Extra: {extra_columns}"
                )

            # Reorder columns to exactly match the model's training order.
            input_df = input_df.reindex(columns=expected_columns)

            st.caption(
                f"Input validation passed: {len(expected_columns)} "
                "features match the trained model."
            )

            probability = float(
                model.predict_proba(input_df)[0, 1]
            )

            threshold = 0.13
            flagged = probability >= threshold

            history_row = {
                **row,
                "prediction_time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "readmission_probability": round(probability, 4),
                "threshold": threshold,
                "model_flag": "Flagged" if flagged else "Not Flagged"
            }

            try:
                save_prediction_history(history_row)
                st.success("Prediction saved to history!")
            except Exception as save_error:
                st.error(
                    f"Could not save prediction history: {save_error}"
                )

            # Display result
            st.divider()
            st.subheader("Prediction Result")
            c1, c2 = st.columns(2)

            c1.metric(
                "Estimated 30-Day Readmission Probability",
                f"{probability:.1%}"
            )
            c2.metric(
                "Experimental Threshold",
                f"{threshold:.0%}"
            )

            if flagged:
                st.warning(
                    "The research model flags this encounter "
                    "above its selected threshold."
                )
            else:
                st.info(
                    "The research model does not flag this encounter "
                    "above its selected threshold."
                )

            st.caption(
                "This experimental result is not a diagnosis or "
                "clinical recommendation. The threshold is not "
                "clinically validated."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ==================================================
# PAGE 2: DASHBOARD
# ==================================================
elif page == "Dashboard":

    st.header("📊 Model Performance Dashboard")
    st.caption(
        "Compare model evaluation metrics and review saved predictions."
    )

    if not RESULTS_PATH.exists():
        st.error(f"Model results file not found: {RESULTS_PATH}")
    else:
        try:
            results = pd.read_csv(RESULTS_PATH)

            required = [
                "Model",
                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Test F1",
                "Test ROC-AUC"
            ]

            missing = [
                col for col in required
                if col not in results.columns
            ]

            if missing:
                st.error(f"Missing results columns: {missing}")
            elif results.empty:
                st.warning("The model comparison file is empty.")
            else:
                best = results.loc[results["Test F1"].idxmax()]

                st.subheader("🏆 Best Model by F1-score")
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Best F1-score Model",
                    str(best["Model"])
                )
                c2.metric("Accuracy", f'{best["Test Accuracy"]:.2%}')
                c3.metric("Precision", f'{best["Test Precision"]:.2%}')
                c4.metric("Recall", f'{best["Test Recall"]:.2%}')
                c5.metric("ROC-AUC", f'{best["Test ROC-AUC"]:.3f}')

                st.divider()
                st.subheader("📈 Model Performance Comparison")

                metric_columns = {
                    "Accuracy": "Test Accuracy",
                    "Precision": "Test Precision",
                    "Recall": "Test Recall",
                    "F1-score": "Test F1",
                    "ROC-AUC": "Test ROC-AUC"
                }

                selected_metric = st.selectbox(
                    "Choose a metric",
                    list(metric_columns.keys()),
                    key="dashboard_metric"
                )

                metric_col = metric_columns[selected_metric]

                st.bar_chart(
                    results.set_index("Model")[[metric_col]],
                    y=metric_col,
                    y_label=selected_metric
                )

                st.subheader("📋 Complete Evaluation Results")
                display_results = results[required].copy()

                for col in [
                    "Test Accuracy",
                    "Test Precision",
                    "Test Recall",
                    "Test F1"
                ]:
                    display_results[col] = display_results[col].map(
                        lambda value: f"{value:.2%}"
                    )

                display_results["Test ROC-AUC"] = (
                    display_results["Test ROC-AUC"].map(
                        lambda value: f"{value:.3f}"
                    )
                )

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "A confusion matrix cannot be reconstructed from "
                    "summary metrics alone. Actual test labels and "
                    "predictions are required."
                )

        except Exception as exc:
            st.error(f"Could not read model comparison results: {exc}")

    # ==================================================
    # PATIENT PREDICTION HISTORY
    # ==================================================
    st.divider()
    st.header("🏥 Patient Prediction History")

    if not HISTORY_PATH.exists():
        st.info(
            "No prediction history found. Select Make Prediction "
            "from the sidebar and submit the patient form first."
        )
    else:
        try:
            history = pd.read_csv(HISTORY_PATH)

            if history.empty:
                st.info("No saved predictions yet.")
            else:
                needed = [
                    "prediction_time",
                    "readmission_probability",
                    "model_flag"
                ]

                missing_history = [
                    col for col in needed
                    if col not in history.columns
                ]

                if missing_history:
                    st.error(
                        f"Missing history columns: {missing_history}"
                    )
                else:
                    history["readmission_probability"] = pd.to_numeric(
                        history["readmission_probability"],
                        errors="coerce"
                    )

                    total = len(history)
                    flagged_count = (
                        history["model_flag"] == "Flagged"
                    ).sum()
                    not_flagged_count = (
                        history["model_flag"] == "Not Flagged"
                    ).sum()

                    latest = history.iloc[-1]
                    latest_probability = latest[
                        "readmission_probability"
                    ]

                    st.subheader("📌 Prediction Summary")
                    p1, p2, p3, p4 = st.columns(4)

                    p1.metric("Total Predictions", total)
                    p2.metric("Flagged", int(flagged_count))
                    p3.metric("Not Flagged", int(not_flagged_count))
                    p4.metric(
                        "Latest Probability",
                        f"{latest_probability:.1%}"
                        if pd.notna(latest_probability)
                        else "N/A"
                    )

                    st.markdown("### 🕒 Latest Prediction")
                    st.write(
                        f'**Prediction Time:** '
                        f'{latest["prediction_time"]}'
                    )
                    st.write(
                        f'**Model Flag:** {latest["model_flag"]}'
                    )
                    st.caption(
                        "Flagged means the estimated probability "
                        "meets or exceeds the experimental 13% "
                        "threshold. It does not mean readmission "
                        "is certain."
                    )

                    st.subheader("📋 All Saved Predictions")
                    filter_col1, filter_col2 = st.columns(2)

                    with filter_col1:
                        selected_flag = st.selectbox(
                            "Filter by model flag",
                            ["All", "Flagged", "Not Flagged"],
                            key="history_flag_filter"
                        )

                    with filter_col2:
                        search_date = st.text_input(
                            "Filter by date (YYYY-MM-DD)",
                            placeholder="e.g. 2026-10-26",
                            key="history_date_filter"
                        )

                    filtered_history = history.copy()

                    if selected_flag != "All":
                        filtered_history = filtered_history[
                            filtered_history["model_flag"] == selected_flag
                        ]

                    if search_date.strip():
                        filtered_history = filtered_history[
                            filtered_history["prediction_time"]
                            .astype(str)
                            .str.contains(search_date.strip(), na=False)
                        ]

                    st.caption(
                        f"Showing {len(filtered_history)} "
                        f"of {len(history)} saved predictions"
                    )

                    history_display = filtered_history[
                        [
                            "prediction_time",
                            "readmission_probability",
                            "model_flag"
                        ]
                    ].copy()

                    history_display["readmission_probability"] = (
                        history_display["readmission_probability"].map(
                            lambda value: f"{value:.1%}"
                            if pd.notna(value)
                            else "N/A"
                        )
                    )

                    st.dataframe(
                        history_display.iloc[::-1],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "⬇️ Export filtered history (CSV)",
                        data=filtered_history.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="hospital_readmission_prediction_history.csv",
                        mime="text/csv"
                    )

                    st.subheader("📉 Prediction Probability Chart")
                    chart_history = filtered_history[
                        ["readmission_probability"]
                    ].dropna().reset_index(drop=True)

                    if not chart_history.empty:
                        chart_history.index = chart_history.index + 1
                        chart_history.index.name = "Prediction Number"

                        st.line_chart(
                            chart_history["readmission_probability"],
                            y_label="Estimated Probability"
                        )
                    else:
                        st.info(
                            "No prediction probabilities available "
                            "for the selected filters."
                        )

                    if st.button("🔄 Refresh Dashboard Data"):
                        st.rerun()

        except Exception as exc:
            st.error(
                f"Could not read prediction history: {exc}"
            )

    st.divider()
    st.caption(
        "Research prototype only. These predictions are experimental "
        "and are not validated for clinical decisions."
    )
