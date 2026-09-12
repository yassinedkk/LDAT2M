"""
This file is provided to you as a skeleton for your implementation of clustering algorithms.
You are not obligated to use it and are free to write any class or method as long as the
following requirements are respected:

Your k_means method must take as parameters:
  - D: numpy array of shape (N, d) containing the data points
  - k: number of clusters
  - epsilon: convergence threshold (default: 1e-6)
  - max_iters: maximum number of iterations (default: 1000)
  - random_seed: random seed for reproducibility (default: 42)

Your dbscan method must take as parameters:
  - D: numpy array of shape (N, d) containing the data points
  - epsilon: float, neighborhood radius
  - MinPts: int, minimum number of points to form a core point

The methods should perform clustering and return the cluster labels.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score


def load_data(filepath):
    """Load data from file. Supports .npy files or text files with space-separated features.
    
    Returns:
        numpy array of shape (N, d)
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    return np.loadtxt(filepath)


def load_solution(filepath):
    """Load solution labels from file. Supports .npy files or text files with one label per line.
    
    Returns:
        numpy array of labels
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    return np.loadtxt(filepath, dtype=int)


def get_ari(predicted_labels, ground_truth_labels):
    """Calculate Adjusted Rand Index score.
    
    Parameters:
        predicted_labels: array of predicted cluster labels
        ground_truth_labels: array of ground truth labels
    
    Returns:
        float: ARI score (ranges from -1 to 1, where 1 is perfect agreement)
    """
    return adjusted_rand_score(ground_truth_labels, predicted_labels)


def k_means(D, k, epsilon=1e-6, max_iters=1000, random_seed=42):
    """K-means clustering algorithm.

    Parameters:
    - D: numpy array of shape (N, d), data points
    - k: int, number of clusters
    - epsilon: float, convergence threshold
    - max_iters: int, maximum iterations
    - random_seed: int, random seed for reproducibility

    Returns:
    - labels: numpy array of shape (N,), cluster assignment for each point
    """
    # TODO
    pass


def dbscan(D, epsilon, MinPts):
    """
    DBSCAN clustering algorithm.

    Parameters:
    - D: numpy array of shape (N, d), data points
    - epsilon: float, neighborhood radius
    - MinPts: int, minimum number of points to form a core point

    Returns:
    - labels: numpy array of shape (N,), cluster assignment for each point
      * -1 indicates noise points (not part of any cluster)
      * 0 to k-1 are cluster IDs
    """
    # TODO
    pass
