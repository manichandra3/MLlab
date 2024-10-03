import numpy as np
import pandas as pd
import sys


def euclideanDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class PartitioningClustering:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.medoids_cost = []

    def initMedoids(self, data):
        self.medoids = []
        indexes = np.random.choice(len(data), self.params["k"], replace=False)
        self.medoids = data[indexes]
        print(f"\nselected medoids >>> {self.medoids}\n")
        self.medoids_cost = [0] * self.params["k"]
        print(f"\nmedoids cost >>> {self.medoids_cost}\n")

    def isConverged(self, new_medoids):
        return np.array_equal(self.medoids, new_medoids)

    def updateMedoids(self, data, labels):
        self.params["has_converged"] = True
        clusters = [[] for _ in range(self.params["k"])]

        for j, label in enumerate(labels):
            clusters[label].append(data[j])

        new_medoids = []
        for i in range(self.params["k"]):
            if clusters[i]:  # check if the cluster is not empty
                new_medoid = clusters[i][0]  # start with the first point
                old_medoid_cost = float('inf')

                for candidate in clusters[i]:
                    cur_medoids_cost = sum(euclideanDistance(candidate, point) for point in clusters[i])
                    if cur_medoids_cost < old_medoid_cost:
                        new_medoid = candidate
                        old_medoid_cost = cur_medoids_cost

                new_medoids.append(new_medoid)

        if not self.isConverged(new_medoids):
            self.medoids = np.array(new_medoids)
            self.params["has_converged"] = False

    def fit(self, data):
        if self.params["method"] == "kmeans":
            self.centroids = {i: data[i] for i in range(self.params["k"])}

            for _ in range(self.params["max_iter"]):
                self.classifications = {i: [] for i in range(self.params["k"])}

                for featureset in data:
                    distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(featureset)

                prev_centroids = self.centroids.copy()
                for classification in self.classifications:
                    if self.classifications[classification]:
                        self.centroids[classification] = np.average(self.classifications[classification], axis=0)

                if all(np.array_equal(prev_centroids[c], self.centroids[c]) for c in self.centroids):
                    break

        elif self.params["method"] == "kmedoids":
            self.initMedoids(data)
            for _ in range(self.params["max_iter"]):
                cur_labels = []
                self.medoids_cost = [0] * self.params["k"]

                for k in range(len(data)):
                    d_list = [euclideanDistance(self.medoids[j], data[k]) for j in range(self.params["k"])]
                    cur_labels.append(np.argmin(d_list))
                    self.medoids_cost[np.argmin(d_list)] += min(d_list)

                print(f"\ntotal medoids cost {self.medoids_cost}")
                self.updateMedoids(data, cur_labels)
                if self.params["has_converged"]:
                    break

            print(f"\nfinal medoids >>> {self.medoids}\n")
            return self.medoids

    def predict(self, data):
        pred = []
        for point in data:
            d_list = [euclideanDistance(medoid, point) for medoid in self.medoids]
            pred.append(np.argmin(d_list))
        return np.array(pred)


if __name__ == "__main__":
    # Load customer data from CSV
    csv_file_path = sys.argv[1]  # CSV file path should be passed as the first argument
    df = pd.read_csv(csv_file_path)

    # Assuming the relevant columns are named as 'Total Spending', 'Number of Transactions', 'Average Purchase Value'
    customer_data = df[['Total Spending', 'Number of Transactions', 'Average Purchase Value']].to_numpy()

    params = {
        'k': int(sys.argv[2]),  # Number of clusters
        'max_iter': 300,
        'has_converged': False,
        'method': sys.argv[3]  # Clustering method ('kmeans' or 'kmedoids')
    }

    p = PartitioningClustering(**params)
    p.fit(customer_data)

    # Assuming you want to predict new data from another CSV file or the same one
    user_data = df[['Total Spending', 'Number of Transactions', 'Average Purchase Value']].to_numpy()
    predictions = p.predict(user_data)

    # Print out the predictions
    for i, pred in enumerate(predictions):
        print(f"Data point {i}: Cluster {pred}")

