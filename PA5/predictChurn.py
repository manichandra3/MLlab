from typing import Tuple, Any
import math
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


def sigmoid(X):
    """
    Returns the sigmoid of a given x
    :param X: x value
    :return: Sigmoid of x
    """
    return 1/(1+math.exp(X)**-1)


def normalize(X: pd.DataFrame) -> Tuple[Any, Any, Any]:
    """
    Normalize the data frame according to the following rules:
    :param X: DataFrame to be normalized
    :return: Tuple containing the normalized DataFrame, mean of each column, and standard deviation of each column
    """
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    X['ContractType'] = X['ContractType'].replace(contract_mapping)

    features = X.drop(columns=['Churn'])

    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    X_norm = (features - mu) / sigma

    X_norm['Churn'] = X['Churn']

    return X_norm, mu, sigma


def logistic_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)

    cost = -np.mean(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))

    return cost


df_normalized, mu, sigma = normalize(df)
print(df_normalized)
print(mu)
