from typing import Tuple, Any

import numpy as np
import pandas as pd
import random

random.seed(42)

data = {
    'Age': [random.randint(18, 70) for _ in range(10)],
    'MonthlyCharges': [round(random.uniform(20.0, 120.0), 2) for _ in range(10)],
    'ContractType': [random.choice(['Month-to-month', 'One year', 'Two year']) for _ in range(10)],
    'Tenure': [random.randint(0, 72) for _ in range(10)],
    'Churn': [random.choice([0, 1]) for _ in range(10)]
}

df = pd.DataFrame(data)


def normalize(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Normalize the data frame according to the following rules:
    :param X: DataFrame to be normalized
    :return: Tuple containing the normalized DataFrame, mean of each column, and standard deviation of each column
    """
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    X['ContractType'] = X['ContractType'].map(contract_mapping)

    features = X.drop(columns=['Churn'])

    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    X_norm = (features - mu) / sigma

    X_norm['Churn'] = X['Churn']

    return X_norm, mu, sigma


def gradient_descent()

df_normalized, mu, sigma = normalize(df)
print(df_normalized)
print(mu)
