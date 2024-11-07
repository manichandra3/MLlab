import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('iris.csv')
data.drop(['Id'], axis=1, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

unique_labels, y_numeric = np.unique(y, return_inverse=True)


def kmeans(X, k, max_iters=1000):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = []
        self.regularization = regularization
        self.lambda_ = lambda_

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        self.weights = np.zeros((len(unique_classes), num_features))
        self.bias = np.zeros(len(unique_classes))

        for idx, cls in enumerate(unique_classes):
            binary_y = np.where(y == cls, 1, 0)
            for _ in range(self.num_iterations):
                linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]
                y_predicted = self.sigmoid(linear_model)

                dw = (1 / num_samples) * np.dot(X.T, (y_predicted - binary_y))
                db = (1 / num_samples) * np.sum(y_predicted - binary_y)

                if self.regularization == 'l2':
                    dw += (self.lambda_ / num_samples) * self.weights[idx]

                self.weights[idx] -= self.learning_rate * dw
                self.bias[idx] -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias
        y_predicted_probs = self.sigmoid(linear_model)
        return np.argmax(y_predicted_probs, axis=1)


def select_closest_samples(X, labels, centroids, num_samples=25):
    closest_samples = []
    for i in range(len(centroids)):
        cluster_samples = X[labels == i]
        cluster_distances = np.linalg.norm(cluster_samples - centroids[i], axis=1)
        closest_indices = np.argsort(cluster_distances)[:num_samples]
        closest_samples.append(cluster_samples[closest_indices])
    return np.vstack(closest_samples)


test_acc = []
ks = [i for i in range(2, 7)]
for k in ks:
    labels, centroids = kmeans(X, k)
    unique_clusters = np.unique(labels)
    print('-----------------------------------------')
    print(f"Number of clusters formed: {len(unique_clusters)}")
    selected_samples = select_closest_samples(X, labels, centroids)
    selected_labels = np.concatenate([y_numeric[labels == i][:25] for i in range(k)])

    train_size = int(0.6 * selected_samples.shape[0])
    X_train = selected_samples[:train_size]
    y_train = selected_labels[:train_size]
    X_val = selected_samples[train_size:]
    y_val = selected_labels[train_size:]

    remaining_indices = np.setdiff1d(np.arange(len(data)),
                                     np.concatenate([np.where(labels == i)[0][:25] for i in range(k)]))
    X_test = X[remaining_indices]
    y_test = y_numeric[remaining_indices]

    model = CustomLogisticRegression(num_iterations=1000, regularization='l2', lambda_=0.1)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print(f'{y_val_pred}vs{y_val}\n')
    val_accuracy = np.mean(y_val_pred == y_val)
    print(f'Validation Accuracy: {val_accuracy}')

    y_test_pred = model.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f'Test Accuracy: {test_accuracy}')
    test_acc.append(test_accuracy)

indexes = [i for i in range(2, 7)]
plt.plot(indexes, test_acc, 'bo-')
plt.ylabel('Number of clusters')
plt.xlabel('Accuracy')
plt.title('Accuracy vs Number of Clusters')
plt.show()
