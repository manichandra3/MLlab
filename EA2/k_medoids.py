
import numpy as np
import sys


def euclideanDistance(x, y):
    squared_d = 0
    for i in range(len(x)):
        squared_d += (x - y) ** 2
    d = np.sqrt(squared_d)
    return d


class PartitioningClustering:
    def __init__(self, **kwargs):
        self.classifications = None
        self.medoids = None
        self.centroids = None
        self.params = kwargs
        self.medoids_cost = []

    def initMedoids(self, data):
        self.medoids = []
        indexes = np.random.randint(0, len(data) - 1, self.params["k"])  # select k indices from all data in dataset
        self.medoids = data[indexes]  # starting medoids(clusters) will be random numbers from dataset
        print(f"\nselected medoids >>> {self.medoids}\n")
        for i in range(0, self.params["k"]):
            self.medoids_cost.append(0)  # for each cluster medoid the cost is 0
        print(f"\nmedoids cost >>> {self.medoids_cost}\n")

    def isConverged(self, new_medoids):  # check that new medoids is equals to the old one - if that case we have to
        # stop the algorithm
        return set([tuple(x) for x in self.medoids]) == set([tuple(x) for x in new_medoids])

    def updateMedoids(self, data, labels):
        self.params["has_converged"] = True
        clusters = []
        for i in range(self.params["k"]):  # storing data points to the current cluster key belong to
            cluster = []
            for j in range(len(data)):
                if labels[j] == i:  # if the label of data is i-th k then the cluster of that data is found
                    cluster.append(data[j])  # data with common labels
            clusters.append(cluster)
            print("\n============================================================")
            print("labels >>> ", labels)
            print("clusters >>> ", clusters)

        new_medoids = []
        for i in range(self.params["k"]):
            new_medoid = self.medoids[i]
            old_medoid_cost = self.medoids_cost[i]
            for j in range(len(clusters[i])):
                cur_medoids_cost = 0
                for dpoint_index in range(len(clusters[i])):
                    cur_medoids_cost += euclideanDistance(clusters[i][j], clusters[i][
                        dpoint_index])  # calculate distance between each point in a cluster to find the new medoid
                if cur_medoids_cost < old_medoid_cost:  # if calculated distance is less than the old one we found a 
                    # new medoid with a minimum cost
                    new_medoid = clusters[i][j]  # update medoid
                    old_medoid_cost = cur_medoids_cost
            new_medoids.append(new_medoid)
            print("\nnew_medoids >>> ", new_medoids)
        if not self.isConverged(new_medoids):
            self.medoids = new_medoids
            self.params["has_converged"] = False

    def fit(self, data):

        if self.params["method"] == "kmeans":
            self.centroids = {}  # dict centroids of each cluster
            for i in range(self.params["k"]):
                self.centroids[i] = data[i]  # set centroids as the number of k - select the k first points of data

            for i in range(self.params["max_iter"]):
                self.classifications = {}  # the result of classification for each cluster
                for i in range(self.params["k"]):
                    self.classifications[
                        i] = []  # number of classifications is based on the number of k which is the number of 
                    # centroids

                for featureset in data:
                    distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in
                                 self.centroids]  # calculate the distance from each centroid for this featureset
                    classification = distances.index(
                        min(distances))  # the index of the minimum distance is the one cluster that this feature 
                    # will be in it
                    self.classifications[classification].append(
                        featureset)  # put the featureset in its cluster - its cluster can be found based on the 
                    # index of the minimum distance from each centroid because there are k centroids then k distances

                    print("--------------classifications--------------")
                    print(self.classifications, "\n")

                prev_centroids = dict(self.centroids)  # save perv centroids
                for classification in self.classifications:
                    self.centroids[classification] = np.average(self.classifications[classification],
                                                                axis=0)  # update each classified centroid based on the average of all data points in that cluster
                    print("--------------centroids--------------")
                    print(self.centroids, "\n")
        elif self.params["method"] == "kmedoids":
            self.initMedoids(data)
            for i in range(self.params["max_iter"]):
                cur_labels = []
                for medoid in range(self.params["k"]):
                    self.medoids_cost[medoid] = 0
                    for k in range(len(data)):
                        d_list = []
                        for j in range(self.params["k"]):
                            d_list.append(euclideanDistance(self.medoids[j], data[k]))
                        cur_labels.append(d_list.index(
                            min(d_list)))  # the index of the minimum distance is the current label for that data - 
                        # because we're calculating k (clusters) distance between each data
                        self.medoids_cost[medoid] += min(
                            d_list)  # the cost of each medoid is the sum of minimum distances in k medoids for each 
                        # data
                        print(f"\nmedoids cost {medoid} = {self.medoids_cost[medoid]}")
                print(f"\ntotal medoids cost {self.medoids_cost}")
                self.updateMedoids(data, cur_labels)  # update medoids at the end of each iteration
                if self.params["has_converged"]:
                    break
            print(f"\nfinal medoids >>> {self.medoids}\n")
            return np.array(self.medoids)

    def predict(self, data):  # cluster new data
        if self.params["method"] == "kmeans":
            distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification
        elif self.params["method"] == "kmedoids":
            pred = []
            for i in range(len(data)):
                d_list = []
                for j in range(len(self.medoids)):
                    d_list.append(euclideanDistance(self.medoids[j], data[
                        i]))  # calculate each distance between each data and found medoids - the index of minimum 
                    # distance is the cluster that the data should be in
                pred.append(d_list.index(min(d_list)))
            return np.array(pred)


if __name__ == "__main__":
    arg_data = [int(a_d) for a_d in sys.argv[1].split(",")]
    input_data = np.array([arg_data]).reshape(-1, 1)
    params = {'k': int(sys.argv[3]), 'max_iter': 300, 'has_converged': False, 'method': sys.argv[2]}

    p = PartitioningClustering(**params)
    p.fit(input_data)
    user_data = np.array([int(a_d) for a_d in input("enter new data to cluster >>>> ").split(",")])
    print(f"CLUSTER IS  |>> {p.predict(user_data)} <<| FOR {user_data}")
