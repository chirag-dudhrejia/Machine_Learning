import random
import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(42)


class Kmeans:
    def __init__(self, clusters=2, max_iter=100):
        self.clusters = clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            # assign clusters  3
            cluster_group = self.assign_clusters(X)
            # move centroids  4
            old_centroids = self.centroids
            self.centroids = self.move_centroids(X, cluster_group)
            print(i)
            # self.ploting(X, cluster_group)
            # check finish  5
            if np.all(old_centroids == self.centroids):
                break
        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distance = []

        for row in X:
            for centroid in self.centroids:
                distance.append(np.sqrt(np.sum(np.square(row-centroid))))
            min_distance = min(distance)
            index_pos = distance.index(min_distance)
            cluster_group.append(index_pos)
            distance.clear()

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for typee in cluster_type:
            new_centroids.append(X[cluster_group == typee].mean(axis=0))

        return np.array(new_centroids)
    
    def ploting(self, X, cluster_group):
        plt.scatter(X[cluster_group == 0, 0], X[cluster_group == 0, 1], color='red')
        plt.scatter(X[cluster_group == 1, 0], X[cluster_group == 1, 1], color='blue')
        plt.scatter(X[cluster_group == 2, 0], X[cluster_group == 2, 1], color='green')
        plt.scatter(X[cluster_group == 3, 0], X[cluster_group == 3, 1], color='yellow')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='X')
        plt.show()
