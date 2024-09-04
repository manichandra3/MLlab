import numpy as np
import pandas as pd
import math

def random_split(data, ratio_x, ratio_v,ratio_y):
    size = data.shape[0]
    size_x = int(size * ratio_x)
    size_v = int(size * ratio_v)
    size_y = int(size * ratio_y)
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)
    indices_y = all_indices[:size_y]
    indices_x = all_indices[size_y:size_v + size_x]
    indices_v = all_indices[size_v + size_x:size_v + size_y + size_x]
    data_x = data.iloc[indices_x]
    data_v = data.iloc[indices_v]
    data_y = data.iloc[indices_y]
    return data_x, data_v,data_y

def MSE(w, x, b, y):
    return np.mean((y-prediction(w, x, b) ** 2))

def prediction(w, x, b):
    return np.dot(w, x) + b

def cost_function(w, x, b, y):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        total_cost += np.square(prediction(w, x[i], b) - y[i])
    return total_cost / (2 * m)


def compute_gradient(w, x, b, y, alpha):
    m = x.shape[0]
    w_cost = np.zeros_like(w)
    b_cost = 0
    for i in range(m):
        error = prediction(w, x[i], b) - y[i]
        w_cost += alpha * x[i] * error / m
        b_cost += alpha * error / m
    return w_cost, b_cost


def gradient_descent(w, x_train, b, y_train, num_iterations, alpha):
    print(f'Gradient descent starts')
    cost_history = []
    b_history = []
    w_history = []
    for i in range(num_iterations):
        w_cost, b_cost = compute_gradient(w, x_train, b, y_train, alpha)
        w -= w_cost
        b -= b_cost
        cost_history.append(cost_function(w, x_train, b, y_train))
        b_history.append(b)
        w_history.append(w)
        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {cost_function(w, x_train, b, y_train)}")
    print(f"{w, b}")
    print(f'Gradient descent ends')
    return w, b, cost_history, w_history, b_history

def validation(x_valid, y_valid):
    J_history = []


# df = pd.read_csv('data.csv')
# train, validation,test = random_split(df, 0.6, 0.2,0.2)
# print(f"training:\n{train}\n\ntesting:\n{test}\n\nvalidation:\n{validation}")
