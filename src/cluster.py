import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def load_data(filename):
    """
    Loads the data from the file
    :param filename: The file name
    :return: The data
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([float(val) for val in line.strip().split('\t')])
    data = np.array(data)
    coordinates, labels = data[:, :2], data[:, 2].astype(np.long)
    return coordinates, labels


def plot_points(cooridnates, labels, title):
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700', '#000000'
    ]
    x, y = zip(*cooridnates)
    x, y = np.array(x), np.array(y)

    plt.figure(figsize=(8, 8))

    for i in np.unique(labels):
        plt.scatter(x[labels == i], y[labels == i], color=colors[i])
    plt.axis('off')
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.close()


def calculate_centroids(coordinates: np.ndarray, clusters: np.ndarray, k: int):
    n = coordinates.shape[0]
    centroids = np.zeros((k, coordinates.shape[1]))
    for i in range(k):
        cluster = coordinates[clusters == i]
        if cluster.shape[0] > 0:
            centroids[i] = np.mean(cluster, axis=0)
        else:
            centroids[i] = coordinates[np.random.randint(0, n)]
    return centroids


def kmeans(coordinates: np.ndarray, k: int, **kwargs):
    # Randomly assign the points to clusters
    n = coordinates.shape[0]
    clusters = np.random.randint(0, k, n)
    print(np.unique(clusters))

    # Calculate the distance between each point and its cluster center
    centroids = calculate_centroids(coordinates, clusters, k)

    # Iterate until the clusters do not change
    for _ in range(100):
        # Assign each point to the closest cluster
        clusters = np.argmin(cdist(coordinates, centroids), axis=1)
        centroids = calculate_centroids(coordinates, clusters, k)
    return clusters


def kernel_kmeans(coordinates: np.ndarray, k: int, kernel: str = 'rbf', sigma: float = 2, **kwargs):
    if kernel == 'rbf':
        K = 1 - np.exp(-cdist(coordinates, coordinates) /
                       (2 * sigma ** 2))
    else:
        raise ValueError

    n = coordinates.shape[0]
    clusters = np.random.randint(0, k, n)

    distance = np.zeros((n, k))
    for i in range(k):
        distance[:, i] = K[:, clusters == i].mean(axis=1)
    clusters = np.argmin(distance, axis=1)
    for i in range(k):
        if i not in np.unique(clusters):
            clusters[np.random.randint(0, n)] = i

    for _ in range(1000):
        for i in range(k):
            distance[:, i] = K[:, clusters == i].mean(axis=1)
        clusters = np.argmin(distance, axis=1)
        for i in range(k):
            if i not in np.unique(clusters):
                clusters[np.random.randint(0, n)] = i
    print(np.unique(clusters))
    return clusters


def ratio_cut(coordinates: np.ndarray, k: int, sigma: float = 0.5, ** kwargs):
    n = coordinates.shape[0]
    W = np.exp(-cdist(coordinates, coordinates) /
               (2 * sigma ** 2)) - np.identity(n)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    vals, vecs = np.linalg.eig(L)
    return kmeans(vecs[:, np.argsort(np.argsort(vals)) < k], k)


def ncut(coordinates: np.ndarray, k: int, sigma: float = 0.7, **kwargs):
    n = coordinates.shape[0]
    W = np.exp(-cdist(coordinates, coordinates) /
               (2 * sigma ** 2)) - np.identity(n)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    vals, vecs = np.linalg.eig(np.linalg.inv(D) @ L)
    return kmeans(vecs[:, np.argsort(np.argsort(vals)) < k], k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--data', type=str,
                        default='cluster_data/Aggregation.txt')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                        choices=['kmeans', 'kernel_kmeans', 'ratio_cut', 'ncut'])
    parser.add_argument('--sigma', type=float, default=0.7)
    args = parser.parse_args()
    # np.random.seed(args.seed)
    dataset = os.path.splitext(os.path.basename(args.data))[0].capitalize()

    coordinates, labels = load_data(args.data)
    num_classes = np.unique(labels).shape[0]
    plot_points(coordinates, labels, dataset)

    cluster_algo = globals()[args.algorithm]

    clusters = cluster_algo(coordinates, num_classes, sigma=args.sigma)
    plot_points(coordinates, clusters, f'{dataset} {args.algorithm}')
