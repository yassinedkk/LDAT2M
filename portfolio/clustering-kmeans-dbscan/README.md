# Clustering with K-means and DBSCAN

Portfolio project for **LINGI2364 - Mining Patterns in Data**. This project
implements two complementary clustering algorithms from scratch: K-means for
compact centroid-based clusters and DBSCAN for density-based clusters with
noise detection.

## Project overview

The implementation includes:

- K-means with Euclidean distance, repeated random initialization, empty-cluster
  handling, and convergence based on centroid displacement;
- DBSCAN with explicit core-point detection, iterative cluster expansion, and
  noise labels;
- loading utilities for text and NumPy datasets;
- evaluation with the Adjusted Rand Index (ARI).

## Usage

```python
from clustering import dbscan, k_means, load_data

points = load_data("datasets/kmeans/kmeans_easy_2.txt")
kmeans_labels = k_means(points, k=4)

points = load_data("datasets/dbscan/moons_shape.txt")
dbscan_labels = dbscan(points, epsilon=0.4, MinPts=5)
```

## Reported results

| Algorithm | Dataset | Parameters | ARI |
| --- | --- | --- | ---: |
| K-means | `kmeans_easy_2` | `k=4` | 1.000 |
| K-means | `kmeans_hard` | `k=4` | 1.000 |
| K-means | `s3` | `k=15` | 0.727 |
| DBSCAN | `moons_shape` | `eps=0.4`, `MinPts=5` | 1.000 |
| DBSCAN | `s2` | `eps=0.1`, `MinPts=10` | 0.707 |
| DBSCAN | `varying_density` | `eps=1.0`, `MinPts=3` | 1.000 |

The full methodology, plots, and discussion are available in `report.pdf`.

## Project structure

- `clustering.py`: submitted K-means and DBSCAN implementation
- `datasets/`: six clustering datasets grouped by algorithm
- `solutions/`: reference cluster labels supplied with the assignment
- `instructions/Instruction.pdf`: original project specification
- `instructions/Instruction.tex`: LaTeX source of the specification
- `starter_template.py`: original starter template
- `requirements.txt`: Python dependencies
- `report.pdf`: anonymized project report

## Author

Yassine Zeamari
