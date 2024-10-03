import numpy as np
import sys
import pandas as pd


def euclideanDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def normalize_features(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_normalized = (x - mean) / std
    return x_normalized, mean, std


class PartitioningClustering:
    def __init__(self, **kwargs):
        self.classifications = None
        self.medoids = None
        self.centroids = None
        self.params = kwargs
        self.medoids_cost = []

    def initMedoids(self, data):
        self.medoids = []
        indexes = np.random.choice(len(data), self.params["k"], replace=False)  # select k unique indices from data
        self.medoids = data[indexes]
        print(f"\nselected medoids >>> {self.medoids}\n")
        self.medoids_cost = [0] * self.params["k"]  # initialize costs
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
    csv_file_path = 'data.csv'
    df = pd.read_csv(csv_file_path)
    customer_data = df[['Total Spending', 'Number of Transactions', 'Average Purchase Value']].to_numpy()
    # Purchase Value
    # normalized_customer_data, mean, std = normalize_features(customer_data)
    k_values = [2, 3, 4, 5]
    params = {'k': k_values[2], 'max_iter': 1000, 'has_converged': False, 'method': 'kmedoids'}
    p = PartitioningClustering(**params)
    p.fit(customer_data)

    user_data = np.array([float(a_d) for a_d in
                          input("Enter new data to cluster (spending, transactions, average value) >>> ").split(",")])
    # normalized_user_data, mn, sd = normalize_features(user_data)
    print(f"\nuser data >>> {user_data}")
    print(f"CLUSTER IS  |>> {p.predict(user_data.reshape(1, -1))} <<| FOR {user_data}")
