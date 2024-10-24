import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
            if clusters[i]:
                new_medoid = clusters[i][0]
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
        if self.params["method"] == "kmedoids":
            self.initMedoids(data)
            for _ in range(self.params["max_iter"]):
                cur_labels = []
                self.medoids_cost = [0] * self.params["k"]
                wcss = 0.0

                for k in range(len(data)):
                    distances = [euclideanDistance(medoid, data[k]) for medoid in self.medoids]
                    label = np.argmin(distances)
                    cur_labels.append(label)
                    self.medoids_cost[label] += distances[label]

                wcss = sum(euclideanDistance(self.medoids[label], data[k]) ** 2 for k, label in enumerate(cur_labels))
                print(f"WCSS for this iteration: {wcss}")

                self.updateMedoids(data, cur_labels)
                if self.params["has_converged"]:
                    break

            print(f"\nfinal medoids >>> {self.medoids}\n")
            return self.medoids, wcss

    def predict(self, data):
        pred = []
        for point in data:
            d_list = [euclideanDistance(medoid, point) for medoid in self.medoids]
            pred.append(np.argmin(d_list))
        return np.array(pred)


def performance_evaluation(k_values, customer_data):
    wcss_values = []
    for k_value in k_values:
        print(f'for k_value {k_value}')
        print('------------------------------------------------')
        params_ = {'k': k_value, 'max_iter': 400, 'has_converged': False, 'method': 'kmedoids'}
        p_ = PartitioningClustering(**params_)
        _, wcss_ = p_.fit(customer_data)
        wcss_values.append(wcss_)
        print('------------------------------------------------')
    return wcss_values


if __name__ == "__main__":
    csv_file_path = 'data.csv'
    df = pd.read_csv(csv_file_path)
    customer_data = df[['Total Spending', 'Number of Transactions', 'Average Purchase Value']].to_numpy()
    normalized_customer_data = normalize_features(customer_data)
    k_values = range(1, 11)
    wcss = performance_evaluation(k_values, customer_data)

    user_data = np.array([float(a_d) for a_d in
                          input("Enter new data to cluster (spending, transactions, average value) >>> ").split(",")])
    print('===========================')
    print(f'prediction for k = 3')
    p = PartitioningClustering(k=3, max_iter=400, has_converged=False, method='kmedoids')
    _, __ = p.fit(customer_data)

    print(f"\nuser data >>> {user_data}")
    print(f"CLUSTER IS  |>> {p.predict(user_data.reshape(1, -1))} <<| FOR {user_data}")
    print('===========================')

    plt.plot(k_values, wcss, '-x', color='red')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters')
    plt.show()
