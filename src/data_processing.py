# src/data_processing.py
# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set a random seed for reproducibility
np.random.seed(0)

# Define N number of samples to simulate number of financial customers
n_samples = 20_000_000

# Create the customer_id array separately and ensure its length matches n_samples
customer_id = np.unique(np.random.randint(1000, 198234870, n_samples))

# If the length of customer_id is less than n_samples, regenerate it
while len(customer_id) < n_samples:
    additional_ids = np.unique(np.random.randint(1000, 198234870, n_samples - len(customer_id)))
    customer_id = np.concatenate((customer_id, additional_ids))

# Create a dictionary for dataset
data = {
    'customer_id': customer_id[:n_samples],  # Trim to exactly n_samples if needed
    'age': np.random.randint(18, 85, n_samples),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
    'dependents': np.random.randint(0, 9, n_samples),
    'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-employed', 'Retired'], n_samples),
    'annual_income': np.random.randint(20_000, 1_000_000, n_samples),
    'loan_amount': np.random.randint(5000, 1_000_000, n_samples),
    'loan_term': np.random.randint(1, 30, n_samples),
    'interest_rate': np.round(np.random.uniform(2.5, 20.0, n_samples), 2),
    'loan_purpose': np.random.choice(['Home', 'Car', 'Education', 'Personal', 'Business'], n_samples),
    'loan_to_value_ratio': np.round(np.random.uniform(0.5, 1.5, n_samples), 2),
    'credit_score': np.random.randint(300, 850, n_samples),
    'debt_to_income_ratio': np.round(np.random.uniform(0.01, 0.5, n_samples), 2),
    'delinquencies': np.random.randint(0, 10, n_samples),
    'credit_history_length': np.random.randint(0, 40, n_samples),
    'default': np.random.randint(0, 2, n_samples),  # 0 for repaid, 1 for default
    'default_amount': np.random.randint(0, 1_000_000, n_samples),  # If default, amount defaulted
    'repayment_tenure': np.random.randint(1, 360, n_samples)  # Number of months before default (if applicable)
}

# Show progress during DataFrame creation
credit_risk_data = pd.DataFrame({key: tqdm(value, desc=key) for key, value in data.items()})

# Create DataFrame from dictionary
# credit_risk_data = pd.DataFrame(data)

# Save dataset to HDD
filepath = "D:\\datasets\\github_credit_risk_modeling_data\\credit_risk_data.csv"

# Save the DataFrame to the processed data folder
credit_risk_data.to_csv(filepath, index=False)

print(f"Data has been generated and saved to {filepath}.")
