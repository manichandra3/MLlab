import numpy as np
import matplotlib.pyplot as plt
from i import random_split

# Y = f(x) = W*X + b
# W = (sum(xy) - sum(x)sum(y)/n) / (sum(x^2) - (sum(x))^2/n)
# b = sum(y)/n - W*sum(x)/n

# data_set = np.random.rand(30, 2)
data_set = np.array([[9.57142857, 6.21903199],
                     [20.71428571, 14.60131255],
                     [16.85714286, 11.6997539],
                     [23.28571429, 16.53568499],
                     [11.28571429, 7.50861362],
                     [24.57142857, 17.50287121],
                     [15.14285714, 10.41017227],
                     [14.71428571, 10.08777687],
                     [17.71428571, 12.34454471],
                     [12.14285714, 8.15340443],
                     [9.14285714, 5.89663659],
                     [5.71428571, 3.31747334],
                     [19.85714286, 13.95652174],
                     [7.42857143, 4.60705496],
                     [5.28571429, 2.99507793],
                     [6.57142857, 3.96226415],
                     [21.57142857, 15.24610336],
                     [23.71428571, 16.85808039],
                     [24.14285714, 17.1804758],
                     [10.42857143, 6.86382281],
                     [7.85714286, 4.92945037],
                     [10.85714286, 7.18621821],
                     [14.28571429, 9.76538146],
                     [18.57142857, 12.98933552],
                     [17.28571429, 12.0221493],
                     [4., 2.02789171],
                     [22.85714286, 16.21328958],
                     [15.57142857, 10.73256768],
                     [18.14285714, 12.66694011],
                     [6.14285714, 3.63986874]])
data1_x, data1_y = random_split(data_set, 0.7, 0.3)
data2_x, data2_y = random_split(data_set, 0.8, 0.2)
data3_x, data3_y = random_split(data_set, 0.9, 0.1)
# print(data1_x)
# Separate the features and target variables
X = data_set[:, 1]
print(f"features: {X}")
Y = data_set[:, 0]
print(f"labels: {Y}")


def prediction(w, x, b):
    return w * x + b


def find_w_and_b(x, y):
    n = x.shape[0]
    numerator = np.sum(x * y) - (np.sum(x) * np.sum(y) / n)
    denominator = np.sum(x ** 2) - (np.sum(x) ** 2) / n
    w = numerator / denominator
    b = np.sum(y) / n - w * np.sum(x) / n
    return w, b


def MSE(x, y, w, b):
    m = x.shape[0]
    return np.mean((prediction(w, x, b) - y) ** 2)


X_train1 = data1_x[:, 1]
Y_train1 = data1_x[:, 0]
X_train2 = data2_x[:, 1]
Y_train2 = data2_x[:, 0]
X_train3 = data3_x[:, 1]
Y_train3 = data3_x[:, 0]
w1, b1 = find_w_and_b(X_train1, Y_train1)
w2, b2 = find_w_and_b(X_train1, Y_train1)
w3, b3 = find_w_and_b(X_train1, Y_train1)
print(f"w: {w1}, b: {b1}")
print(f"w: {w2}, b: {b2}")
print(f"w: {w3}, b: {b3}")

X_test1 = data1_y[:, 1]
Y_test1 = data1_y[:, 0]
X_test2 = data2_y[:, 1]
Y_test2 = data2_y[:, 0]
X_test3 = data3_y[:, 1]
Y_test3 = data3_y[:, 0]
err1 = MSE(X_test1, Y_test1, w1, b1)
err2 = MSE(X_test2, Y_test2, w2, b2)
err3 = MSE(X_test3, Y_test3, w3, b3)
print(f"Mean Squared Error(70:30): {err1}")
print(f"Mean Squared Error(80:20): {err2}")
print(f"Mean Squared Error(90:10): {err3}")

plt.scatter(X, Y, color='blue', label='Data')
plt.plot(X, prediction(w1, X, b1), color='red', label='Regression line(70:30)')
plt.plot(X, prediction(w2, X, b2), color='green', label='Regression line(80:20)')
plt.plot(X, prediction(w3, X, b3), color='yellow', label='Regression line(90:10)')
plt.xlabel('Number of hours of sunshine')
plt.ylabel('Number of ice creams sold')
plt.title('Linear Regression')
plt.legend()
plt.show()
