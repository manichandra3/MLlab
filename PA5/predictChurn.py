from copy import deepcopy
from typing import Tuple, Any
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

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
    return 1 / (1 + np.exp(-X))

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

def logistic_gradient(X, y, w, b):
    m, n = X.shape
    f_wb = sigmoid(np.dot(X, w) + b)
    err = f_wb - y
    dj_dw = np.dot(X.T, err)
    dj_db = np.sum(err)
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = logistic_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(logistic_cost(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history

def predict(X, w, b):
    """
    Predicts the class labels for the given input data.
    :param X: Input feature data
    :param w: Weights
    :param b: Bias
    :return: Predicted class labels
    """
    probabilities = sigmoid(np.dot(X, w) + b)
    return (probabilities >= 0.5).astype(int)

df_normalized, mu, sigma = normalize(df)

features = df_normalized.drop(columns=['Churn']).values
target = df_normalized['Churn'].values
w_tmp = np.zeros(features.shape[1])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out, J_hist = gradient_descent(features, target, w_tmp, b_tmp, alph, iters)
print(f"\nUpdated parameters: w:{w_out}, b:{b_out}")

predictions = predict(features, w_out, b_out)
df['PredictedChurn'] = predictions
print("Predictions:")
print(df[['Churn', 'PredictedChurn']])

plt.plot(J_hist)
plt.show()
