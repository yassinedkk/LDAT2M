# 3D Gesture Recognition: DTW vs Deep Learning

This project compares a classical Dynamic Time Warping (DTW) + k-nearest-neighbors baseline with a feed-forward neural network for classifying 3D hand trajectories.

Two gesture domains are studied:

- **Domain 1:** ten digit gestures;
- **Domain 4:** ten three-dimensional shapes, including cones, cylinders, spheres, pyramids, and toroids.

Each domain contains 2,000 trajectory files: 10 subjects × 10 gesture classes × 10 repetitions. Every observation records the hand position `(x, y, z)` through time.

## Methods

- trajectory-level centering and standardization;
- PCA projection from three spatial coordinates to two components;
- temporal interpolation to a fixed length of 85 points;
- DTW distance with a window constraint and weighted 3-NN classification;
- neural network with 128- and 64-unit hidden layers, ReLU activations, and dropout;
- 10-fold user-dependent and user-independent evaluation;
- confusion matrices and per-class classification reports.

## Main results

| Domain | Method | User-dependent | User-independent |
|---|---|---:|---:|
| Digit gestures | DTW + 3-NN | 97.7% | 96.8% |
| Digit gestures | Neural network | **100.0%** | **99.3%** |
| 3D shapes | DTW + 3-NN | 98.7% | 88.6% |
| 3D shapes | Neural network | **99.7%** | **97.9%** |

The neural network achieved the strongest mean accuracy in every setting and generalized substantially better to unseen users on the more difficult 3D-shape domain.

## Repository structure

```text
portfolio/gesture-recognition/
├── README.md
├── analysis.ipynb
├── data/
│   ├── LICENSE
│   ├── README.txt
│   ├── domain1_subjects_1-5.zip
│   ├── domain1_subjects_6-10.zip
│   ├── domain4_subjects_1-5.zip
│   └── domain4_subjects_6-10.zip
├── report.pdf
└── requirements.txt
```

The dataset is split into four archives only to keep individual repository files compact. Extract all four archives into `data/` before running the notebook.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip "data/domain1_subjects_*.zip" -d data
unzip "data/domain4_subjects_*.zip" -d data
jupyter lab analysis.ipynb
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` and extract the four ZIP files using File Explorer or `Expand-Archive`.

## Authors

- Gaetan Berlaimont
- Benoit Henrion
- Yassine Zeamari

Group project for LINFO2275, *Data Mining and Decision Making*, UCLouvain (2025).

## Dataset source and license

The gesture data are described by Huang, Jaiswal & Rai (2019), “Gesture-based system for next generation natural and intuitive interfaces.” The supplied dataset distribution includes the GNU General Public License v3; its original `LICENSE` and `README.txt` files are preserved in `data/`.
