import pandas as pd
import numpy as np

data = pd.read_csv('iris.csv')

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
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


k = 3
labels, centroids = kmeans(X, k)
unique_clusters = np.unique(labels)
print(f"Number of clusters formed: {len(unique_clusters)}")


def select_closest_samples(X, labels, centroids, num_samples=25):
    closest_samples = []
    for i in range(len(centroids)):
        cluster_samples = X[labels == i]
        cluster_distances = np.linalg.norm(cluster_samples - centroids[i], axis=1)
        closest_indices = np.argsort(cluster_distances)[:num_samples]
        closest_samples.append(cluster_samples[closest_indices])
    return np.vstack(closest_samples)


selected_samples = select_closest_samples(X, labels, centroids)
selected_labels = np.concatenate([y_numeric[labels == i][:25] for i in range(k)])

train_size = int(0.7 * selected_samples.shape[0])
X_train = selected_samples[:train_size]
y_train = selected_labels[:train_size]
X_val = selected_samples[train_size:]
y_val = selected_labels[train_size:]

remaining_indices = np.setdiff1d(np.arange(len(data)),
                                 np.concatenate([np.where(labels == i)[0][:25] for i in range(k)]))
X_test = X[remaining_indices]
y_test = y_numeric[remaining_indices]

model = CustomLogisticRegression(num_iterations=1000)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
print(f'{y_val_pred}vs{y_val}\n')
val_accuracy = np.mean(y_val_pred == y_val)
print(f'Validation Accuracy: {val_accuracy}')

y_test_pred = model.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
print(f'Test Accuracy: {test_accuracy}')
