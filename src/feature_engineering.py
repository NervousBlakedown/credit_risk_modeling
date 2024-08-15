# src/feature_engineering.py
# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset from Feather file
print("Loading dataset...")
df = pd.read_feather('D:\\datasets\\github_credit_risk_modeling_data\\credit_risk_data.feather')

# Feature Engineering
# 1. Credit-to-Income Ratio
print("Creating Credit-to-Income Ratio...")
df['credit_to_income_ratio'] = df['loan_amount'] / df['income']

# 2. Age Buckets (did quartiles instead) (e.g., 18-30: Young, 31-50: Middle-aged, 51+: Senior)
print("Creating Age Buckets...")
bins = [18, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']
df['age_bucket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# 3. Log Transform of Loan Amount to handle skewed distribution
print("Applying Log Transform to Loan Amount...")
df['log_loan_amount'] = np.log1p(df['loan_amount'])  # log1p is log(1 + x), handles 0s

# 4. Interaction Terms (e.g., age and income)
print("Creating Interaction Terms...")
df['age_income_interaction'] = df['age'] * df['income']

# 5. Scaling Features (if needed)
print("Scaling Features...")
scaler = StandardScaler()
df[['age', 'income', 'loan_amount', 'credit_to_income_ratio']] = scaler.fit_transform(
    df[['age', 'income', 'loan_amount', 'credit_to_income_ratio']]
)

# 6. Handle Missing Values (if any)
print("Handling Missing Values...")
df.fillna(0, inplace=True)  # Example: filling NaNs with 0

# Save the engineered dataset
output_path = 'D:\\datasets\\github_credit_risk_modeling_data\\engineered_credit_risk_data.feather'
df.to_feather(output_path)

print(f"Feature engineering complete. Data saved to {output_path}.")
