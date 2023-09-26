import pandas as pd
import numpy as np

# For verfications.
import scipy.spatial.distance
import scipy.cluster.hierarchy
import sklearn.cluster

class DistanceMatrix(object):
    def __init__(self):
        self.matrix = dict()

    def __setitem__(self, key, value):
        i, j = key
        if i > j:
            i, j = j, i
        self.matrix[i, j] = value
        
    def __getitem__(self, key):
        i, j = key
        if i == j:
            return 0
        if i > j:
            i, j = j, i
        return self.matrix[i, j]

euclidean = lambda x, y: np.sqrt(np.sum((x - y)*(x - y)))

def ward(x, y, i, D, C):
    t = 1.0 / (C[x] + C[y] + C[i])
    return np.sqrt((C[x] + C[i]) * t * D[x, i] * D[x, i] +
                (C[y] + C[i]) * t * D[y, i] * D[y, i] -
                C[i] * t * D[x, y] * D[x, y])
    
def main():
    df = pd.read_csv('data/Golub_dataset_train.csv', index_col=0, header=[0, 1])
    X = df.values

    # Distance matrix between vectors.
    D = DistanceMatrix()
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i < j:
                D[i, j] = euclidean(X[i, :], X[j, :])

    # Sizes of clusters.
    C = dict()
    for i in range(X.shape[0]):
        C[i] = 1

    # Table of linkages between clusters.
    Z = np.zeros((X.shape[0] - 1, 4))
    for k in range(X.shape[0] - 1):
        # Find the two clusters in the closest neighborhood.
        minimum = np.Infinity
        for i in C.keys():
            for j in C.keys():
                if i < j and D[i, j] < minimum:
                    minimum = D[i, j]
                    x, y = i, j

        # Create the new cluster from x and y.
        C[X.shape[0] + k] = C[x] + C[y]
        
        # Record the new cluster.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = D[x, y]
        Z[k, 3] = C[X.shape[0] + k]
        
        # Update the distance matrix.
        for i in C.keys():
            if i < X.shape[0] + k:
                D[i, X.shape[0] + k] = ward(x, y, i, D, C)
                    
        # Clusters x and y are included in the new cluster.
        del C[x], C[y]
        
    # Sort Z by cluster distances.
    Z = Z[np.argsort(Z[:, 2])]


    # Verification against Scipy and Scikit-learn.
    assert np.allclose(Z, scipy.cluster.hierarchy.ward(scipy.spatial.distance.pdist(df)))

    model = sklearn.cluster.AgglomerativeClustering(metric='euclidean', linkage='ward', compute_distances=True).fit(df)
    assert np.allclose(Z[:, :3], np.column_stack([model.children_, model.distances_]))

    print(Z)

if __name__ == '__main__':
    main()
