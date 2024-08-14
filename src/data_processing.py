# src/data_processing.py
# Imports
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(0)

# Define the number of samples
n_samples = 10_000

# Create a dictionary with data
data = {
    'customer_id': np.unique(np.random.randint(1000, 198234870, n_samples)),
    'age': np.random.randint(18, 85, n_samples),
    'income': np.random.randint(20_000, 1_000_000, n_samples),
    'loan_amount': np.random.randint(5000, 50_000, n_samples),
    'default': np.random.randint(0, 2, n_samples)  # 0 for repaid, 1 for default
}

# Create a DataFrame
credit_risk_data = pd.DataFrame(data)

# Save the DataFrame to the processed data folder
credit_risk_data.to_csv('../data/processed/credit_risk_data.csv', index=False)

print("Data has been generated and saved to ../data/processed/credit_risk_data.csv")
