import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plt_regression(w, b, x):
    return w * x + b


def prediction(w, x, b):
    return np.dot(w, x) + b


def cost_function(w, x, b, y):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        total_cost += np.square(prediction(w, x[i], b) - y[i])
    return total_cost / (2 * m)


def compute_gradient(w, x, b, y, alpha):
    error = prediction(w, x, b) - y
    w_cost = alpha * x * error
    b_cost = alpha * error
    return w_cost, b_cost


def stochastic_gradient_descent(w, x, b, y, num_iterations=50, alpha=0.01):
    print(f'Stochastic gradient descent starts')
    cost_history = []
    b_history = []
    w_history = []
    m = x.shape[0]

    for i in range(num_iterations):
        permutation = np.random.permutation(m)
        x_shuffled = x[permutation]
        y_shuffled = y[permutation]
        for j in range(m):
            w_cost, b_cost = compute_gradient(w, x_shuffled[j], b, y_shuffled[j], alpha)
            w -= w_cost
            b -= b_cost
        cost_history.append(cost_function(w, x, b, y))
        b_history.append(b)
        w_history.append(w)
        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {cost_function(w, x, b, y)}")
    print(f"{w,b}")
    print(f'Stochastic gradient descent ends')
    return w, b, cost_history, w_history, b_history


df = pd.read_csv('data.csv', delimiter=',')
print(df.shape)
with pd.option_context('display.max_columns', 40):
    print(df.describe(include='all'))

x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values

x_train = x_train.reshape(-1, 1)
w = np.zeros(x_train.shape[1])
b = 0

w_final, b_final, cost_plt, w_plt, b_plt = stochastic_gradient_descent(w, x_train, b, y_train)
print(f"Final weights: {w_final}, Final bias: {b_final}")

plt.title("Regression Plot")
plt.plot(x_train, plt_regression(w_final, b_final, x_train), color='orange', label="Prediction")
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], label="Actual Data")
plt.xlabel('Exam Score')
plt.ylabel('Number of Hours of Study')
plt.legend()
plt.show()

plt.title("Cost Function Plot")
plt.plot(cost_plt, label="Cost")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.title("Cost vs. Weights Plot")
plt.plot(w_plt, cost_plt, label="Cost vs. Weights")
plt.xlabel('Weights')
plt.ylabel('Cost')
plt.legend()
plt.show()
