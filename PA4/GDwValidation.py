import sys

import numpy as np
import pandas as pd


def random_split(data, ratio_x, ratio_v, ratio_y):
    size = data.shape[0]
    size_x = int(size * ratio_x)
    size_v = int(size * ratio_v)
    size_y = int(size * ratio_y)
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)
    indices_y = all_indices[:size_y]
    indices_v = all_indices[size_y:size_y + size_v]
    indices_x = all_indices[size_y + size_v:]
    data_x = data.iloc[indices_x]
    data_v = data.iloc[indices_v]
    data_y = data.iloc[indices_y]
    return data_x, data_v, data_y


def MSE(w, x, b, y):
    return np.mean((y - prediction(w, x, b)) ** 2)


def prediction(w, x, b):
    return np.dot(x, w) + b


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
    print('Gradient descent starts')
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
    print('Gradient descent ends')
    return w, b, cost_history, w_history, b_history


def k_fold_cross_validation(k, x, y, num_iterations, alpha, ratio_x, ratio_v, ratio_y):
    fold_size = len(x) // k
    avg_cost = 0
    min_cost = sys.maxsize
    b_best = 0
    w_best = 0

    for fold in range(k):
        print(f"Fold {fold + 1} starts")

        start_val = fold * fold_size
        end_val = start_val + fold_size

        x_val = x[start_val:end_val]
        y_val = y[start_val:end_val]

        x_train = np.concatenate((x[:start_val], x[end_val:]), axis=0)
        y_train = np.concatenate((y[:start_val], y[end_val:]), axis=0)

        df_train = pd.DataFrame({'x': x_train, 'y': y_train})
        df_train_x, df_train_v, df_train_y = random_split(df_train, ratio_x, ratio_v, ratio_y)

        x_train = df_train_x['x'].values.reshape(-1, 1)
        y_train = df_train_x['y'].values
        x_val = df_train_v['x'].values.reshape(-1, 1)
        y_val = df_train_v['y'].values

        w = np.zeros(x_train.shape[1])
        b = 0

        w, b, cost_history, _, _ = gradient_descent(w, x_train, b, y_train, num_iterations, alpha)

        val_cost = MSE(w, x_val, b, y_val)
        if val_cost < min_cost:
            min_cost = val_cost
            b_best = b
            w_best = w
        print(f"Fold {fold + 1}: Validation Cost = {val_cost}")

        avg_cost += val_cost

    avg_cost /= k
    print(f"Average Validation Cost after {k}-Fold Cross Validation: {avg_cost}")
    print(f"best: w: {w_best}, b: {b_best}")
    return avg_cost


df = pd.read_csv('data.csv')

x = np.array(df['Hours of Study'])
y = np.array(df['Exam Score'])

k_fold_cross_validation(5, x=x, y=y, num_iterations=10, alpha=0.001, ratio_x=0.7, ratio_v=0.2, ratio_y=0.1)
