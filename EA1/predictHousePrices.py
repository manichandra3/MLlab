import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict(w, x, b):
    return np.dot(x, w) + b


def normalize_features(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_normalized = (x - mean) / std
    return x_normalized, mean, std


def compute_gradient(w, x, b, y):
    m = x.shape[0]
    predictions = predict(w, x, b)
    error = predictions - y
    w_gradient = np.dot(x.T, error) / m
    b_gradient = np.sum(error) / m
    return w_gradient, b_gradient


def cost_function(w, x, b, y):
    m = x.shape[0]
    predictions = predict(w, x, b)
    squared_errors = np.square(predictions - y)
    total_cost = np.sum(squared_errors) / (2 * m)
    return total_cost


def gradient_descent(w, x, b, y, iterations, alpha):
    J_history = []
    b_history = []
    w_history = []
    m = x.shape[0]
    for i in range(iterations):
        w_gradient, b_gradient = compute_gradient(w, x, b, y)
        w -= alpha * w_gradient
        b -= alpha * b_gradient
        J_history.append(cost_function(w, x, b, y))
        w_history.append(w.copy())
        b_history.append(b)
        if i % 10 == 0:
            print(f'Iteration {i}, Cost {J_history[-1]}, W: {w}, B: {b}')
    return w_history, b_history, J_history, w, b


df = pd.read_csv('data.csv')
x_train, mean, std = normalize_features(df.iloc[:, :-1].values)
y_train = df.iloc[:, -1].values
w_init = np.zeros(x_train.shape[1])
b_init = 0
iters = 100
alpha = 0.1
w_hist, b_hist, J_hist, w_final, b_final = gradient_descent(w_init, x_train, b_init, y_train, iters, alpha)

print('Final W:', w_final)
print('Final B:', b_final)

x_predict = np.array([2500, 4, 10, 5])
x_predict_normalized = (x_predict - mean) / std
print('Prediction:', predict(w_final, x_predict_normalized, b_final))

plt.plot(range(iters), J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.show()
