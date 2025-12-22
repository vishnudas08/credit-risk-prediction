# Credit Risk Prediction System – End-to-End ML Risk Modeling & Deployment

 ## Project Overview
This project implements a production-oriented Credit Risk Prediction System designed to assess the likelihood of loan default using historical applicant and loan data. The system combines advanced feature engineering, class imbalance handling, multiple ML model comparisons, business impact analysis, and real-time deployment through a Streamlit interface.

The solution is built to go beyond model accuracy by translating predictions into financial risk insights, enabling data-driven lending decisions.

# System Objectives
- Predict loan default probability accurately
- Handle highly imbalanced datasets
- Compare classical and ensemble ML models
- Optimize model performance using ROC-AUC
- Convert ML predictions into business value
- Deploy a real-time inference UI without heavy backend infrastructure


##System Architecture
#Architecture Overview
The system follows a modular ML architecture:

1. Data Processing Layer
Data loading, validation, and cleaning
Outlier handling and missing value imputation

2. Feature Engineering Layer
Financial risk indicators
Behavioral and demographic features
Encoded categorical variables

3. Modeling Layer
SMOTE-based imbalance correction
Multiple classifier training
Cross-validation and hyperparameter tuning

4. Evaluation & Business Layer
ML metrics (ROC-AUC, precision, recall)
Financial impact analysis

5. Deployment Layer
Streamlit-based UI
Real-time prediction using saved model artifacts

# Hyperparameter Optimization
For the best-performing model (XGBoost):
Applied GridSearchCV
Tuned:
Tree depth
Learning rate
Number of estimators
Minimum child weight
This improved generalization and reduced overfitting.

# Business Impact Analysis

Rather than stopping at ML metrics, the system evaluates financial outcomes:

Metrics Calculated

Correctly identified defaults

Missed defaults

False rejections

Estimated loan losses

Prevented financial losses

Opportunity cost

Net business benefit





## Real-Time Deployment
Deployment Strategy

Built a Streamlit UI for live predictions

Loaded trained model, scaler, and feature schema using joblib

Enforced strict feature ordering to avoid silent inference errors

Returned:

Default risk classification

Probability of default

Why No Heavy Backend?

Streamlit executes Python natively

Ideal for prototypes, demos, and PoCs

Faster iteration and lower operational complexity

#Key Challenges & Solutions
Challenge	Solution
Class imbalance	SMOTE oversampling
Feature mismatch at inference	Saved feature schema
Silent preprocessing errors	Strict input validation
Business interpretation	Financial impact metrics
# Key Learnings
Feature engineering often outweighs model complexity
ROC-AUC is more reliable than accuracy in risk modeling
Class imbalance must be handled explicitly
ML success includes deployment and business reasoning
Small pipeline inconsistencies can silently break systems

# Future Enhancements
Threshold tuning for risk appetite optimization
SHAP-based explainability
REST API using FastAPI
Cloud deployment (GCP / Streamlit Cloud)
Power BI integration for portfolio-level insights

## Conclusion
This project demonstrates a production-grade ML workflow with strong emphasis on:
Financial risk modeling
Explainability
Business impact
Real-time usability

The Credit Risk Prediction System showcases how machine learning can be responsibly applied to high-stakes financial decision-making with transparency and measurable value.

