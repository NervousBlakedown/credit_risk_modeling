# src/model_building.py
# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the engineered dataset from Feather file
print("Loading engineered dataset...")
df = pd.read_feather('D:\\datasets\\github_credit_risk_modeling_data\\credit_risk_data.feather')

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# One-Hot Encode categorical variables
print("One-Hot Encoding categorical variables...")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target variable
print("Defining features and target variable...")
X = df.drop(columns=['customer_id', 'default'])  # Drop non-feature columns
y = df['default']  # Target variable

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a model (e.g., Logistic Regression)
print("Training Logistic Regression model...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Evaluate the model
print("Evaluating Logistic Regression model...")
y_pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print(f"ROC AUC Score: {roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]):.4f}")

# Save the Logistic Regression model
logreg_model_path = 'D:\\datasets\\github_credit_risk_modeling_data\\logreg_model.joblib'
joblib.dump(logreg, logreg_model_path)
print(f"Logistic Regression model saved to {logreg_model_path}")

# Train and evaluate a Random Forest model as well
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the Random Forest model
print("Evaluating Random Forest model...")
y_pred_rf = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(f"ROC AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

# Save the Random Forest model
rf_model_path = 'D:\\datasets\\github_credit_risk_modeling_data\\rf_model.joblib'
joblib.dump(rf, rf_model_path)
print(f"Random Forest model saved to {rf_model_path}")

print("Model building complete.")
