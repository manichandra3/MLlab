from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import GD
import SGD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
x_train = df.iloc[:, 0].values
y_train = df.iloc[:, 1].values

x_train = x_train.reshape(-1, 1)
w_gd = np.zeros(x_train.shape[1])
b_gd = 0
w_sgd = np.zeros(x_train.shape[1])
b_sgd = 0
w_final_gd, b_final_gd, cost_plt_gd, w_plt_gd, b_plt_gd = GD.gradient_descent(w_gd,
                                                                              x_train,
                                                                              b_gd,
                                                                              y_train)
w_final_sgd, b_final_sgd, cost_plt_sgd, w_plt_sgd, b_plt_sgd = SGD.stochastic_gradient_descent(w_sgd,
                                                                                               x_train,
                                                                                               b_sgd,
                                                                                               y_train)

plt.title("Regression Plot")
plt.plot(x_train, GD.plt_regression(w_final_gd, b_final_gd, x_train),
         color='red', label="GD Prediction")
plt.plot(x_train, SGD.plt_regression(w_final_sgd, b_final_sgd, x_train),
         color='green', label="SGD Prediction")
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], label="Actual Data")
plt.xlabel('Exam Score')
plt.ylabel('Number of Hours of Study')
plt.legend()
plt.show()

plt.title("Cost Function Plot")
plt.plot(cost_plt_gd, label="GD Cost plot")
plt.plot(cost_plt_sgd, label="SGD Cost plot")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.show()
