import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KMeansClustering:
    def __init__(self, k, max_iters=1000, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Initialize centroids using k-means++ method
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            # Assign points to the nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0)
                                      if np.sum(self.labels == i) > 0
                                      else self.centroids[i]
                                      for i in range(self.k)])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.labels, self.centroids

    def _initialize_centroids(self, X):
        """Initialize centroids using k-means++ method"""
        centroids = [X[np.random.randint(X.shape[0])]]

        for _ in range(1, self.k):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
            centroids.append(next_centroid)

        return np.array(centroids)


class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization='l2',
                 lambda_=0.01, batch_size=32, early_stopping_rounds=5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow

    def _get_batch(self, X, y, batch_size):
        indices = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[batch_indices], y[batch_indices]

    def fit(self, X, y, X_val=None, y_val=None):
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        # Initialize weights and bias
        self.weights = np.zeros((num_classes, num_features))
        self.bias = np.zeros(num_classes)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_iterations):
            # Mini-batch training
            for batch_X, batch_y in self._get_batch(X, y, self.batch_size):
                # One-hot encode batch_y
                batch_y_onehot = np.eye(num_classes)[batch_y]

                # Forward pass
                logits = np.dot(batch_X, self.weights.T) + self.bias
                probas = self.sigmoid(logits)

                # Compute gradients
                error = probas - batch_y_onehot
                dw = (1 / len(batch_X)) * np.dot(error.T, batch_X)
                db = (1 / len(batch_X)) * np.sum(error, axis=0)

                # Add regularization if specified
                if self.regularization == 'l2':
                    dw += (self.lambda_ / len(batch_X)) * self.weights

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Early stopping check
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_rounds:
                        break

    def _compute_loss(self, X, y):
        probas = self.predict_proba(X)
        y_onehot = np.eye(len(self.classes_))[y]
        return -np.mean(y_onehot * np.log(probas + 1e-10))

    def predict_proba(self, X):
        logits = np.dot(X, self.weights.T) + self.bias
        return self.sigmoid(logits)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def evaluate_model(y_true, y_pred, classes):
    """Calculate and return various metrics for model evaluation"""
    accuracy = np.mean(y_true == y_pred)

    # Calculate per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(classes):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return accuracy, class_metrics


def main():
    # Load and preprocess data
    data = pd.read_csv('iris.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert labels to numeric
    unique_labels, y_numeric = np.unique(y, return_inverse=True)

    results = []
    for k in range(2, 7):
        # Perform k-means clustering
        kmeans = KMeansClustering(k=k)
        labels, centroids = kmeans.fit(X_scaled)

        # Select samples closest to centroids
        selected_samples = []
        selected_labels = []
        samples_per_cluster = 25

        for i in range(k):
            cluster_mask = labels == i
            cluster_samples = X_scaled[cluster_mask]
            cluster_labels = y_numeric[cluster_mask]

            if len(cluster_samples) > 0:
                distances = np.linalg.norm(cluster_samples - centroids[i], axis=1)
                closest_indices = np.argsort(distances)[:samples_per_cluster]
                selected_samples.append(cluster_samples[closest_indices])
                selected_labels.append(cluster_labels[closest_indices])

        selected_samples = np.vstack(selected_samples)
        selected_labels = np.concatenate(selected_labels)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            selected_samples, selected_labels, test_size=0.4, random_state=42
        )

        # Train model
        model = MultiClassLogisticRegression(
            learning_rate=0.01,
            num_iterations=1000,
            regularization='l2',
            lambda_=0.1,
            batch_size=32,
            early_stopping_rounds=5
        )
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        remaining_mask = ~np.in1d(np.arange(len(X_scaled)),
                                  np.concatenate([np.where(labels == i)[0][:samples_per_cluster]
                                                  for i in range(k)]))
        X_test = X_scaled[remaining_mask]
        y_test = y_numeric[remaining_mask]

        y_pred = model.predict(X_test)
        accuracy, class_metrics = evaluate_model(y_test, y_pred, unique_labels)

        results.append({
            'k': k,
            'accuracy': accuracy,
            'class_metrics': class_metrics
        })

        print(f"\nResults for k={k}:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print("\nPer-class metrics:")
        for class_name, metrics in class_metrics.items():
            print(f"\n{class_name}:")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    ks = [result['k'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    plt.plot(ks, accuracies, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Number of Clusters')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
