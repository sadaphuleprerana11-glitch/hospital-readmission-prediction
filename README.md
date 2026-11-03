# 🏥 Hospital Readmission Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the likelihood of hospital readmission within 30 days using Machine Learning techniques.

The project uses the **Diabetes 130-US Hospitals Dataset (1999–2008)** from the UCI Machine Learning Repository, containing 101,766 hospital encounter records.

The objective is to explore patient-related information, develop predictive models, compare their performance, and demonstrate predictions through an interactive Streamlit application.

## 🎯 Objectives

- Analyze hospital encounter data using Exploratory Data Analysis (EDA).
- Preprocess clinical and demographic features.
- Develop machine learning classification models.
- Compare model performance using multiple evaluation metrics.
- Build an interactive Streamlit prediction application.

## 📊 Dataset

- Dataset: Diabetes 130-US Hospitals for Years 1999–2008
- Records: 101,766 hospital encounters
- Hospitals: 130
- Prediction target: 30-day hospital readmission
- Positive class: 11,357 (11.16%)
- Negative class: 90,409 (88.84%)

Dataset source: [UCI Machine Learning Repository](https://doi.org/10.24432/C5230J)

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 🤖 Machine Learning Models

- Logistic Regression
- Random Forest
- Gradient Boosting

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 63.85% | 15.28% | 52.53% | 23.68% | 0.6242 |
| Random Forest | 62.86% | 15.75% | 57.00% | 24.68% | 0.6436 |
| Gradient Boosting | 74.32% | 18.82% | 42.45% | 26.08% | 0.6619 |

Gradient Boosting achieved the highest F1-score among the three evaluated models.

## 💻 Application Features

- Patient information input interface
- 30-day readmission probability prediction
- Threshold-based prediction flag
- Model performance dashboard
- Prediction history visualization

## ⚙️ Project Setup

Clone the repository:

```bash
git clone https://github.com/sadaphuleprerana11-glitch/hospital-readmission-prediction.git
```

Navigate to the project directory:

```bash
cd hospital-readmission-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

**Note:** The trained model file (`hospital_readmission_model.joblib`) must also be available in the project directory before running the application.

## 🔮 Future Scope

- Evaluate models using recent hospital datasets.
- Explore advanced machine learning techniques.
- Integrate explainable AI methods such as SHAP.
- Improve prediction performance and model calibration.
- Conduct external validation.

## ⚠️ Disclaimer

This project is an academic research prototype. The model and its experimental prediction threshold are not clinically validated. Predictions should not be used for patient-care decisions.

## 👩‍💻 Author

**Prerana Sadaphule**

GitHub: [sadaphuleprerana11-glitch](https://github.com/sadaphuleprerana11-glitch)
