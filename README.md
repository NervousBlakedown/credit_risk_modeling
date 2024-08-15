# credit_risk_modeling
A project to evaluate and predict credit risk using machine learning models, with a focus on feature engineering and model optimization

## Overview

This project aims to evaluate and predict credit risk using machine learning models. The primary goal is to build a model that can assess the likelihood of an individual defaulting on a loan based on features like age, income, and loan amount. This project demonstrates the application of data science and machine learning techniques to a common problem in finance.

## Installation

To replicate this project, you'll need to clone the repository and install the required Python packages. You can do this as follows:

'''bash
git clone https://github.com/yourusername/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling
pip install -r requirements.txt'''

## Data
The dataset used in this project is synthetic and generated within the code. It includes the following features:

customer_id: Unique identifier for each customer.
age: Age of the customer.
income: Annual income of the customer.
loan_amount: The amount of loan taken.
default: Binary variable indicating whether the customer defaulted (1) or repaid the loan (0).

## Methodology
Exploratory Data Analysis (EDA): Understanding the dataset, identifying patterns, and detecting anomalies.
Data Preprocessing and Feature Engineering: Cleaning the data and creating meaningful features.
Model Building: Training machine learning models to predict credit risk.
Model Tuning and Evaluation: Optimizing models and evaluating their performance using metrics like accuracy, precision, recall, and AUC.

## Visualizations
The project includes various visualizations to better understand the data and model performance. For instance:

3D Scatter Plot: Visualizes the relationship between age, income, loan amount, and default status.
Confusion Matrix and Classification Report: Used to evaluate the model's predictions.

## Usage
To run the analysis, start by exploring the data in the 01_eda.ipynb notebook. Then, proceed to feature engineering, model training, and tuning as per the notebooks in the notebooks/ directory.

Alternatively, you can execute the entire pipeline using the Python scripts in the src/ directory.

## Results
The final model's performance is evaluated using various metrics, and the results are summarized in the reports/ directory.

## Future Work
Incorporate additional features like credit history and debt-to-income ratio.
Experiment with more advanced models like Random Forests, Gradient Boosting, and Neural Networks.
Deploy the model using Flask or Django to create a web-based credit risk assessment tool.

## Contributing
Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

