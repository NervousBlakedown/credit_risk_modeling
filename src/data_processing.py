import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Start the timer (data creation purposes only)
start_time = time.time()

# Set a random seed for reproducibility
np.random.seed(0)

# Define N number of samples to simulate the number of financial customers
n_samples = 20_000_000

# Create the customer_id array separately and ensure its length matches n_samples
customer_id = np.unique(np.random.randint(1000, 198234870, n_samples))

# If the length of customer_id is less than n_samples, regenerate it
while len(customer_id) < n_samples:
    additional_ids = np.unique(np.random.randint(1000, 198234870, n_samples - len(customer_id)))
    customer_id = np.concatenate((customer_id, additional_ids))

# Generate structured random data
data = {
    'customer_id': customer_id[:n_samples],
    'age': np.random.randint(18, 85, n_samples),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
    'dependents': np.random.randint(0, 5, n_samples),
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
}

# Introduce a probabilistic relationship between income, loan amount, credit score, and default
default_prob = (
    (data['loan_amount'] / data['annual_income']) * 0.5 + 
    (700 - data['credit_score']) * 0.001 +
    data['debt_to_income_ratio'] * 2
)
default_prob = np.clip(default_prob, 0, 1)  # Ensure probabilities are between 0 and 1

# Generate the default column based on the probability
data['default'] = np.random.binomial(1, default_prob, n_samples)

# Generate the default_amount based on whether the customer defaulted
data['default_amount'] = np.where(data['default'] == 1, np.random.randint(0, 1_000_000, n_samples), 0)

# Add repayment_tenure based on default and loan term
data['repayment_tenure'] = np.where(data['default'] == 1, np.random.randint(1, 360, n_samples), data['loan_term'] * 12)

# Create DataFrame
credit_risk_data = pd.DataFrame({key: tqdm(value, desc=key) for key, value in data.items()})

# Save dataset to HDD
filepath = "D:\\datasets\\github_credit_risk_modeling_data\\credit_risk_data.csv"
credit_risk_data.to_csv(filepath, index=False)

# End the timer (data creation purposes only)
end_time = time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to hours, minutes, and seconds
hours = int(elapsed_time // 3600) # integer division to get whole hours (not 'true division')
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

# Print the time in a readable format
if hours > 0:
    print(f"Data generation and saving completed in {hours} hours, {minutes} minutes, and {seconds:.2f} seconds.")
elif minutes > 0:
    print(f"Data generation and saving completed in {minutes} minutes and {seconds:.2f} seconds.")
else:
    print(f"Data generation and saving completed in {seconds:.2f} seconds.")

print(f"Data has been generated and saved to {filepath}.")
