"""
Clustering algorithms.
These algorithms will be used for clustering features/attributes.
Three different algorithms:
    1. Fuzzy K-Means (baseline)
    2. Fuzzy spectral clustering
    3. HDBSSCAN
"""

import numpy as np
import time, skfuzzy
from abc import ABC, abstractmethod
import copy
from minepy import MINE
from deap.benchmarks import *
from networkx.convert_matrix import *
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp

class Cluster(ABC):
    '''
    Abstract class for the various clustering techniques
    test
    '''

    def __init__(self, data):
        self.data = data
        super().__init__()

    @abstractmethod
    def assign_clusters(self):
        '''
        Abstract method to assign data points to clusters
        implementation varies by method. Returns the cluster
        assignments as an array.
        '''

        pass

class Kmeans(Cluster):
    '''
    Implementation of the fuzzy k-means clustering algorithm using
    the scikit learn fuzzy implementation
    '''

    def __init__(self, data, k=3):
        self.k = k
        super().__init__(data)

    def assign_clusters(self):
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(self.data.T, self.k, 2, error=0.005, maxiter=1000,
                                                            init=None)
        u_pred, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans_predict(self.data.T, cntr, 2, error=0.005, maxiter=1000)
        return u_pred

class HDbscan(Cluster):
    '''
    Wrapper for the scitkit learn implementation of DBSCAN
    '''

    def __init__(self, data, min_points=4, e=0.5):
        self.min_points = min_points
        self.e = e
        super().__init__(data)

    def assign_clusters(self):
        '''
        Assign the datapoints to clusters using Soft HDBSCAN
        and return an array of the cluster assignments
        '''

        pass

class Spectral(Cluster):
    '''
    Sprectral Clustering
    '''

    def __init__(self, IM, num_clusters):
        super().__init__(IM)
        self.IM = IM
        self.k = num_clusters
        self.IM_graph = nx.to_networkx_graph(self.IM, create_using=nx.Graph)
        self.k_means_clusters = None

    def assign_clusters(self):
        '''
        Assign the datapoints to clusters using spectral clustering and return and array of cluster assignemnts
        '''

        # get Laplacian
        self.laplacian = nx.linalg.laplacian_matrix(self.IM_graph)
        self.laplacian = sp.sparse.csr_matrix.toarray(self.laplacian)
        # print('laplacian', self.laplacian)

        # calc eigen vectors and values of the laplacian
        self.eig_values, self.eig_vectors = np.linalg.eig(self.laplacian)
        sorted_indices = self.eig_values.argsort()
        self.eig_values = self.eig_values[sorted_indices]
        self.eig_vectors = self.eig_vectors[sorted_indices]

        # take k largest eigen vectors
        k_arr = np.arange(self.k)
        self.eig_values = self.eig_values[k_arr]
        self.eig_vectors = np.transpose(self.eig_vectors[k_arr])

        # run fuzzy kmeans with the eigen vectors
        self.k_means_clusters = Kmeans(self.eig_vectors, self.k).assign_clusters()
        print(self.k_means_clusters)

    def return_factors(self, threshold= 0.75):
        factors = []
        outliers = np.arange(len(self.k_means_clusters[0,:]))
        for i in range(self.k):
            cluster = []
            for j in range(len(self.k_means_clusters[i,:])):
                if self.k_means_clusters[i,j] > threshold:
                    cluster.append(j)

            factors.append(cluster)
        outliers = set(outliers) - set.union(*map(set, factors))
        factors.append(list(outliers))


        return factors

if __name__ == '__main__':
    pass