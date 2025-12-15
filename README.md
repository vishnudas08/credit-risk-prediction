# Credit Risk Prediction using Logistic Regression

## 📌 Project Overview
This project predicts whether a loan applicant is likely to default or not based on
demographic and financial attributes. It demonstrates a complete machine learning
pipeline including data cleaning, feature engineering, scaling, and model evaluation.

## 📊 Dataset
Features used:
- person_age
- person_income
- person_emp_length
- loan_amnt
- loan_percent_income

Target:
- loan_status (0 = No Default, 1 = Default)

## ⚙️ Workflow
1. Data exploration and validation
2. Domain-based cleaning (age & employment length)
3. Outlier handling using quantiles
4. Feature scaling using StandardScaler
5. Logistic Regression model training
6. Model evaluation using accuracy and classification report

## 🧠 Model Used
- Logistic Regression (scikit-learn)

## 📈 Results
- Achieved ~85% accuracy on test data
- Model provides interpretable predictions for credit approval decisions

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
# credit-risk-prediction
