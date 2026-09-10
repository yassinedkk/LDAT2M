# Insurance Claim Frequency Modeling

An end-to-end actuarial data-science project for predicting motor-insurance claim frequency and segmenting policyholders. The work compares a Poisson GLM benchmark with neural networks, explains predictions with model-interpretation methods, and explores customer profiles through representation learning and clustering.

## Project scope

- **Supervised learning:** Poisson GLM and four Keras neural-network architectures.
- **Evaluation:** Poisson deviance on chronological-independent train/validation splits.
- **Calibration:** global bias correction and subgroup comparisons.
- **Interpretability:** partial-dependence plots, ICE, LIME, and SHAP.
- **Unsupervised learning:** autoencoders, variational autoencoders, and K-means clustering.

## Selected results

| Model | Training deviance | Validation deviance |
|---|---:|---:|
| Poisson GLM | 0.6665 | **0.6238** |
| Neural network 1 | **0.6623** | 0.6280 |
| Neural network 2 | 0.6827 | 0.6411 |
| Neural network 3 | 0.6863 | 0.6463 |
| Neural network 4 | 0.6984 | 0.6552 |

The simplest neural network produced the best neural result, while the GLM remained slightly stronger on validation data. K-means selected two clusters with distinct portfolio profiles, though its claim-frequency deviance (0.6754) was weaker than the supervised models.

## Repository structure

```text
portfolio/insurance-claim-frequency/
├── README.md
├── analysis.ipynb
├── data/
│   └── train_contrats_anonymized.csv
├── report.pdf
└── requirements.txt
```

The public dataset excludes the original contract number, vehicle-registration identifier, and exported row index. The report has also been stripped of the student number.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab analysis.ipynb
```

On Windows PowerShell, activate the environment with `.venv\\Scripts\\Activate.ps1`.

## Author

Yassine Zeamari — academic project for LDATS2310, *Data Science for Finance and Insurance*, UCLouvain (2026).

This project is for educational and portfolio purposes.
