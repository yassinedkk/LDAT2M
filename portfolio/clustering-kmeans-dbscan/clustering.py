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

from matplotlib.pylab import Generator
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


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
    
    rand=np.random.default_rng(random_seed)
    D = np.asarray(D, dtype=float)
    n,d=D.shape
     # Store best clustering result
    final_labels = None
    min_error = np.inf
     # Multiple random initializations
    for _ in range(20):
        init_centre=rand.choice(n,size=k,replace=False) # Random centroid initialization

        u=D[init_centre].copy()
        t=0
            # Main K-means loop
        while t< max_iters:
             t+=1
             u_prev=u.copy()
             C=[[] for i in range(k)]
       # Assign each point to nearest centroid
             for j in range(n):
               x=D[j]

               dist=np.sum((x-u)**2,axis=1)
               i=np.argmin(dist)

               C[i].append(j)
        # Update centroids
             for i in range(k):
               
               if len(C[i]) > 0:
                 u[i] = np.mean(D[C[i]], axis=0)
               else:
                 u[i] = D[rand.integers(n)] # Handle empty clusters

             diff = np.sum(np.sum((u - u_prev) ** 2, axis=1))
             if diff <= epsilon:# Convergence criterion
                    break
       # Convert clusters into labels
        labels = np.empty(n, dtype=int)

        for i in range(k):
                for l in C[i]:
                    labels[l] = i

        error = np.sum((D - u[labels]) ** 2)
        # Keep best solution
        if error < min_error:
            min_error = error
            final_labels = labels.copy()

    return final_labels

    

def get_neighbors(D, p, epsilon):
    # Find all points within epsilon distance from point p
    return list(np.where(np.sqrt(np.sum((D - D[p]) ** 2, axis=1)) <= epsilon)[0])

def core_objects(D, p, epsilon, MinPts):
    # Check if point p is a core point
    return len(get_neighbors(D, p, epsilon)) >= MinPts

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
    if np.max(D) >500:# Standardize data if values are very large
      D = (D - D.mean(axis=0)) / D.std(axis=0)
    n = D.shape[0]
      # -2 = unvisited, -1 = noise, 0,1,... = cluster labels
    labels = np.full(n, -2, dtype=int)

    index = 0
     # Continue while there are unvisited points
    while np.any(labels == -2):

        unvisited = np.where(labels == -2)[0]
        p = unvisited[0]

        labels[p] = -1
         # If p is a core point, start a new cluster
        if core_objects(D, p, epsilon, MinPts):
            N = get_neighbors(D, p, epsilon)

            labels[p] = index

            k = 0
            while k < len(N):# Expand the cluster

                p_prime = N[k]
               # If neighbor is unvisited, add it to cluster
                if labels[p_prime] == -2:

                    labels[p_prime] = index
                       # If neighbor is also core, add its neighbors
                    if core_objects(D, p_prime, epsilon, MinPts):
                        N_prime = get_neighbors(D, p_prime, epsilon)
                        N.extend(N_prime)



                k += 1

            index += 1

    return labels

