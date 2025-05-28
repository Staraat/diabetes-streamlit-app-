# Diabetes Risk Prediction & Glycemic Load Estimator

This project includes:
- A machine learning pipeline to train, tune, and interpret models for diabetes risk prediction.
- A deployed Streamlit app offering interactive prediction, SHAP explanations, and a Glycemic Index (GI) & Glycemic Load (GL) calculator.

---

## Objective

To predict diabetes risk using behavioral and health indicators, and help users understand food-related blood sugar impact through GI & GL analysis.

---

## Tools Used

- Python (pandas, matplotlib, scikit-learn, XGBoost, Random Forest, Logistic Regression, SHAP)
- Streamlit
- Jupyter/Google Colab
- Excel (GI/GL dataset)
- SHAP for explainability

---

## ML Pipeline Overview

1. **Exploratory Data Analysis**  
   Health survey dataset cleaned and visualized

2. **Model Training**  
   - Logistic Regression (baseline + tuned with GridSearchCV)
   - Random Forest (base + tuned with RandomizedSearchCV)
   - XGBoost (base + final with interaction features)

3. **Evaluation Metrics**  
   Accuracy, Macro F1, Recall, Precision, ROC AUC, Confusion Matrix

4. **Interpretability**  
   SHAP TreeExplainer for feature-level explanations

5. **Final Model**  
   `XGBoost` with engineered features selected based on best performance and interpretability

---

## Streamlit App Features

[🔗 Live Demo (hosted on Streamlit Cloud)]([https://diabxgb.streamlit.app/])

1. **Diabetes Risk Predictor**  
   Users can input health info and lifestyle factors to get a predicted risk level

2. **SHAP Explanation**  
   Displays the top contributing factors increasing and reducing risk

3. **GI & GL Calculator**  
   Lookup GI/GL values for foods, add them to a meal basket, and get smart dietary suggestions

4. **Personalized Recommendations**  
   Tailored health and nutrition advice based on user inputs

---

## Model Comparison Summary
XGBoost is the best model compared to Random Forest and Logistic Regression in this case.
The results was evaluated by Classification Report and Confusion Metrix.

*Final model integrated into Streamlit app*

---

## What I Learned

- How to tune models with Grid/RandomizedSearchCV
- Handling class imbalance and evaluating multiclass metrics
- Deploying interpretable ML models with SHAP
- Using Streamlit to build user-centric, health-focused tools

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It does **not** constitute medical advice or diagnosis. Always consult a qualified healthcare provider.


