import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ClusteringParams:
    k: int
    max_iter: int = 400
    method: str = 'kmedoids'
    tol: float = 1e-4


class PartitioningClustering:
    """
    Implementation of K-Medoids clustering algorithm with improved efficiency and robustness.
    """

    def __init__(self, params: ClusteringParams):
        self.params = params
        self.medoids: Optional[np.ndarray] = None
        self.inertia_: float = float('inf')
        self.n_iter_: int = 0

    def _compute_distances(self, data: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between data points and medoids."""
        distances = np.zeros((len(data), self.params.k))
        for i, medoid in enumerate(self.medoids):
            distances[:, i] = np.sqrt(np.sum((data - medoid) ** 2, axis=1))
        return distances

    def _init_medoids(self, data: np.ndarray) -> None:
        """Initialize medoids using k-means++ initialization strategy."""
        n_samples = len(data)
        # Choose first medoid randomly
        self.medoids = np.zeros((self.params.k, data.shape[1]))
        first_idx = np.random.randint(n_samples)
        self.medoids[0] = data[first_idx]

        # Choose remaining medoids
        for i in range(1, self.params.k):
            distances = self._compute_distances(data)
            min_distances = np.min(distances[:, :i], axis=1)
            probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
            next_idx = np.random.choice(n_samples, p=probabilities)
            self.medoids[i] = data[next_idx]

    def _update_medoids(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Update medoids positions using vectorized operations."""
        new_medoids = np.zeros_like(self.medoids)
        has_changed = False

        for i in range(self.params.k):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                continue

            # Compute total distances for each point in the cluster
            distances = np.zeros(len(cluster_points))
            for j, point in enumerate(cluster_points):
                distances[j] = np.sum(np.sqrt(np.sum((cluster_points - point) ** 2, axis=1)))

            # Select point with minimum total distance as new medoid
            new_medoid_idx = np.argmin(distances)
            new_medoids[i] = cluster_points[new_medoid_idx]

            if not np.array_equal(new_medoids[i], self.medoids[i]):
                has_changed = True

        return new_medoids, has_changed

    def fit(self, data: np.ndarray) -> 'PartitioningClustering':
        """
        Fit the K-Medoids model to the data.

        Args:
            data: numpy array of shape (n_samples, n_features)

        Returns:
            self: fitted model
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self._init_medoids(data)

        for iteration in range(self.params.max_iter):
            # Assign points to clusters
            distances = self._compute_distances(data)
            labels = np.argmin(distances, axis=1)

            # Update medoids
            new_medoids, has_changed = self._update_medoids(data, labels)

            # Compute inertia (WCSS)
            self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)

            if not has_changed:
                break

            self.medoids = new_medoids
            self.n_iter_ = iteration + 1

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        distances = self._compute_distances(data)
        return np.argmin(distances, axis=1)


def normalize_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features using z-score standardization."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_normalized = (x - mean) / std
    return x_normalized, mean, std


def perform_elbow_analysis(data: np.ndarray, k_range: range) -> List[float]:
    """Perform elbow analysis for different k values."""
    wcss_values = []

    for k in k_range:
        print(f'Analyzing k={k}...')
        params = ClusteringParams(k=k)
        model = PartitioningClustering(params)
        model.fit(data)
        wcss_values.append(model.inertia_)
        print(f'WCSS for k={k}: {model.inertia_:.2f}')

    return wcss_values


if __name__ == "__main__":
    # Load and prepare data
    csv_file_path = 'data.csv'
    df = pd.read_csv(csv_file_path)
    features = ['Total Spending', 'Number of Transactions', 'Average Purchase Value']
    customer_data = df[features].to_numpy()

    # Normalize data
    normalized_data, mean, std = normalize_features(customer_data)

    # Perform elbow analysis
    k_range = range(1, 11)
    wcss_values = perform_elbow_analysis(normalized_data, k_range)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss_values, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    # Get user input and make prediction
    try:
        user_input = input("Enter new data to cluster (spending, transactions, average value) >>> ")
        user_data = np.array([float(x) for x in user_input.split(',')])

        # Normalize user data using same parameters as training data
        user_data_normalized = (user_data - mean) / std

        # Fit model with k=3 and predict
        model = PartitioningClustering(ClusteringParams(k=3))
        model.fit(normalized_data)
        cluster = model.predict(user_data_normalized)

        print('\n=== Prediction Results ===')
        print(f'Input data: {user_data}')
        print(f'Assigned cluster: {cluster[0]}')
        print(f'Number of iterations: {model.n_iter_}')
        print(f'Final WCSS: {model.inertia_:.2f}')

    except ValueError:
        print("Error: Please enter three numeric values separated by commas")