


import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Readmission Prediction")
st.subheader("Model Evaluation & Patient Prediction Dashboard")
st.caption("MCA Research Project | Machine Learning")

BASE_DIR = Path(__file__).parent
RESULTS_PATH = BASE_DIR / "model_comparison_results.csv"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"

# ==================================================
# SECTION 1: MODEL EVALUATION
# ==================================================
st.header("📊 Model Evaluation Dashboard")

if not RESULTS_PATH.exists():
    st.error("model_comparison_results.csv was not found.")
    st.stop()

results = pd.read_csv(RESULTS_PATH)

required = [
    "Model", "Test Accuracy", "Test Precision",
    "Test Recall", "Test F1", "Test ROC-AUC"
]

missing = [col for col in required if col not in results.columns]

if missing:
    st.error(f"Missing columns in results CSV: {missing}")
    st.stop()

best = results.loc[results["Test F1"].idxmax()]

st.markdown("### 🏆 Best Overall Model (by F1-score)")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Selected Model", str(best["Model"]))
c2.metric("Accuracy", f'{best["Test Accuracy"]:.2%}')
c3.metric("Precision", f'{best["Test Precision"]:.2%}')
c4.metric("Recall", f'{best["Test Recall"]:.2%}')
c5.metric("ROC-AUC", f'{best["Test ROC-AUC"]:.3f}')

st.divider()

st.markdown("### 📈 Model Performance Comparison")

metric_columns = {
    "Accuracy": "Test Accuracy",
    "Precision": "Test Precision",
    "Recall": "Test Recall",
    "F1-score": "Test F1",
    "ROC-AUC": "Test ROC-AUC"
}

selected_metric = st.selectbox(
    "Choose a performance metric",
    list(metric_columns.keys())
)

metric_col = metric_columns[selected_metric]
chart = results.set_index("Model")[[metric_col]]

st.bar_chart(chart, y=metric_col, y_label=selected_metric)

st.markdown("### 📋 Complete Evaluation Results")

display_results = results[required].copy()

for col in [
    "Test Accuracy", "Test Precision",
    "Test Recall", "Test F1"
]:
    display_results[col] = display_results[col].map(
        lambda x: f"{x:.2%}"
    )

display_results["Test ROC-AUC"] = display_results["Test ROC-AUC"].map(
    lambda x: f"{x:.3f}"
)

st.dataframe(display_results, use_container_width=True)

st.info(
    "A confusion matrix cannot be reconstructed from summary metrics alone. "
    "Actual test labels and model predictions are required."
)

st.divider()

# ==================================================
# SECTION 2: PATIENT PREDICTION HISTORY
# ==================================================
st.header("🏥 Patient Prediction History")

if not HISTORY_PATH.exists():
    st.info(
        "No prediction history found yet. Open the prediction app, "
        "submit the patient form, and click Predict Readmission Risk."
    )
else:
    try:
        history = pd.read_csv(HISTORY_PATH)

        if history.empty:
            st.info("No patient predictions have been saved yet.")
        else:
            required_history = [
                "prediction_time",
                "readmission_probability",
                "model_flag"
            ]

            missing_history = [
                col for col in required_history
                if col not in history.columns
            ]

            if missing_history:
                st.error(
                    f"Prediction history is missing columns: "
                    f"{missing_history}"
                )
            else:
                history["readmission_probability"] = pd.to_numeric(
                    history["readmission_probability"],
                    errors="coerce"
                )

                total_predictions = len(history)
                flagged_count = (
                    history["model_flag"] == "Flagged"
                ).sum()
                not_flagged_count = (
                    history["model_flag"] == "Not Flagged"
                ).sum()

                latest = history.iloc[-1]

                st.markdown("### 📌 Prediction Summary")

                p1, p2, p3, p4 = st.columns(4)

                p1.metric("Total Predictions", total_predictions)
                p2.metric("Flagged", int(flagged_count))
                p3.metric("Not Flagged", int(not_flagged_count))

                latest_probability = latest[
                    "readmission_probability"
                ]

                p4.metric(
                    "Latest Estimated Probability",
                    f"{latest_probability:.1%}"
                    if pd.notna(latest_probability)
                    else "N/A"
                )

                st.markdown("### 🕒 Latest Prediction")

                st.write(
                    f'**Prediction Time:** {latest["prediction_time"]}'
                )
                st.write(
                    f'**Model Flag:** {latest["model_flag"]}'
                )

                st.caption(
                    "Flagged means the estimated probability meets or "
                    "exceeds the experimental 0.13 threshold. It does "
                    "not mean readmission is certain."
                )

                st.markdown("### 📋 All Saved Predictions")

                # Show summary fields only, avoiding unnecessary
                # display of demographic and medication details.
                history_display = history[
                    [
                        "prediction_time",
                        "readmission_probability",
                        "model_flag"
                    ]
                ].copy()

                history_display["readmission_probability"] = (
                    history_display["readmission_probability"].map(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    )
                )

                history_display = history_display.iloc[::-1]

                st.dataframe(
                    history_display,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### 📊 Prediction Probability Chart")

                chart_history = history[
                    ["prediction_time", "readmission_probability"]
                ].copy()

                chart_history = chart_history.dropna(
                    subset=["readmission_probability"]
                )

                if not chart_history.empty:
                    chart_history = chart_history.reset_index(drop=True)
                    chart_history.index = chart_history.index + 1
                    chart_history.index.name = "Prediction Number"

                    st.line_chart(
                        chart_history["readmission_probability"],
                        y_label="Estimated Probability"
                    )

                if st.button("🔄 Refresh Prediction History"):
                    st.rerun()

    except Exception as exc:
        st.error(f"Could not read prediction history: {exc}")

# ==================================================
# DISCLAIMER
# ==================================================
st.divider()

st.caption(
    "Research prototype only. Predictions are experimental, "
    "are based on historical data, and are not validated for "
    "clinical decisions."
)
