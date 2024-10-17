import numpy as np
from sklearn.datasets.samples_generator import make_blobs

X_train, Y_train = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=6, random_state=11)


def normalize(X):
    x1_min = min(X_train[:, 0])
    x1_max = max(X_train[:, 0])

    f = lambda x: (x - x1_min) / (x1_max - x1_min)
    X[:, 0] = f(X[:, 0])

    x2_min = min(X_train[:, 1])
    x2_max = max(X_train[:, 1])

    f = lambda x: (x - x2_min) / (x2_max - x2_min)
    X[:, 1] = f(X[:, 1])

    return X


X = normalize(X_train)
print(X[0:5])


def find_neighbors(k, X_tr, new_point):
    neighbor_arr = []
    for i in range(len(X_tr)):
        dist = np.sqrt(sum(np.square(X_tr[i] - new_point)))
        neighbor_arr.append([i, dist])
    neighbor_arr = sorted(neighbor_arr, key=lambda x: x[1])

    return neighbor_arr[0:k]


from collections import Counter


def classifier(neighbor_arr):
    class_arr = [Y_train[i[0]] for i in neighbor_arr]
    return Counter(class_arr).most_common(1)[0][0]


new_points = np.array([[-10, -10],
                       [0, 10],
                       [-15, 10],
                       [5, -2]])

new_points = normalize(new_points)

knn = find_neighbors(4, X, new_points[1])
classifier(knn)

from sklearn.datasets.samples_generator import make_regression

X_train, Y_train = make_regression(n_samples=300, n_features=2, n_informative=2, noise=5, bias=30, random_state=200)

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c="red", alpha=.5, marker='o')
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('Y')
plt.show()
