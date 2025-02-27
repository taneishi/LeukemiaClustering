import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# For plotting.
import scipy.cluster

def plot_dendrogram(df, Z):
    n_samples = df.shape[0]

    fig = plt.figure(figsize=(5, 5))
    gs = plt.GridSpec(1, 30, wspace=0.3)
    ax_dendrogram = fig.add_subplot(gs[0, 0:29])
    ax_label = fig.add_subplot(gs[0, 29])

    scipy.cluster.hierarchy.set_link_color_palette(['orange', 'green', 'red'])

    dendrogram = scipy.cluster.hierarchy.dendrogram(Z, labels=list(df.index),
            color_threshold=70, above_threshold_color='gray',
            orientation='left', leaf_font_size=9, leaf_rotation=0, ax=ax_dendrogram)

    for spine in ax_label.spines:
        ax_label.spines[spine].set_visible(False)

    ax_label.set_xticks([])
    ax_label.set_ylim(ax_dendrogram.get_ylim())
    ax_label.set_yticks(ax_dendrogram.get_yticks())
    ax_label.yaxis.tick_right()
    ax_label.tick_params(bottom=False, right=False)

    ax_label.set_yticklabels(ax_dendrogram.yaxis.get_ticklabels())
    ax_dendrogram.set_yticks([])

    ymin, ymax = ax_dendrogram.get_ylim()
    for i, label in enumerate(ax_label.yaxis.get_ticklabels()):
        if label.get_text().startswith('ALL.T-cell'):
            color = 'orange'
        elif label.get_text().startswith('ALL.B-cell'):
            color = 'red'
        elif label.get_text().startswith('AML'):
            color = 'green'
        ax_label.add_patch(Rectangle((0, (ymax / n_samples) * i), 1.0, ymax / n_samples, fc=color))
        label.set_fontsize(8)

    ax_dendrogram.axline([70, 0], [70, 1], linestyle='--', linewidth=0.8, color='gray')
    plt.savefig('figure/Golub_Clustering.png', dpi=100, bbox_inches='tight')

def ward(x, y, i, D, C):
    t = 1.0 / (C[x] + C[y] + C[i])
    return np.sqrt((C[x] + C[i]) * t * D[x, i] * D[x, i] +
                (C[y] + C[i]) * t * D[y, i] * D[y, i] -
                C[i] * t * D[x, y] * D[x, y])

euclidean = lambda x, y: np.sqrt(np.sum((x - y)*(x - y)))

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

def clustering(X):
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
        # Find the two nearest clusters.
        minimum = np.Infinity
        for i in C.keys():
            for j in C.keys():
                if i < j and D[i, j] < minimum:
                    minimum = D[i, j]
                    x, y = i, j

        # Label a new cluster compatible with sklearn and scipy.
        new_cluster = X.shape[0] + k

        # Create a new cluster from x and y.
        C[new_cluster] = C[x] + C[y]

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

    return Z

def main():
    train = pd.read_csv('data/Golub.csv', index_col=0, header=[0, 1])

    print('Training set shape', train.shape)

    print('Label breakdown', train.groupby(train.index.str[:3]).size().to_dict())

    X = train.values

    Z = clustering(X)

    print(Z)

    plot_dendrogram(train, Z)

if __name__ == '__main__':
    main()
