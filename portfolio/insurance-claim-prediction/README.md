# Insurance Claim Prediction

An end-to-end binary classification project that predicts whether an automobile insurance policy will generate a claim during its coverage period. The work was completed for the LDATS2350 Data Mining course and has been reorganized into a reproducible, portfolio-ready analysis.

## Project overview

The target, `claimNumbMD`, indicates whether a claim occurred. The analysis covers exploratory data analysis, duplicate and outlier checks, preprocessing, model comparison, and interpretation of the selected model.

Five classifiers are compared:

- Logistic regression
- Decision tree
- K-nearest neighbours
- Multilayer perceptron
- Gaussian Naive Bayes

The original study found logistic regression to offer the strongest balance of discrimination and interpretability, with a reported test ROC AUC of approximately **0.69**. Naive Bayes achieved the highest recall for claims (about **0.81**) but produced substantially more false positives.

## Repository structure

```text
portfolio/insurance-claim-prediction/
├── README.md
├── analysis.qmd
├── requirements.txt
└── .gitignore
```

The dataset is not committed because it was not included in the original archive. Place the course dataset at `data/dataSetJune2025.csv` before rendering the analysis.

## Reproduce the analysis

Requirements: Python 3.10+ and [Quarto](https://quarto.org/).

```bash
cd portfolio/insurance-claim-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
quarto render analysis.qmd
```

## Methodology

The portfolio version improves the original notebook in several ways:

- all preprocessing is learned on the training data through scikit-learn pipelines, preventing train-test leakage;
- the dataset path is portable rather than tied to one computer;
- every model uses the same stratified split and evaluation functions;
- model selection emphasizes ROC AUC, recall, precision, F1 score, and operational trade-offs rather than accuracy alone;
- random seeds are fixed for reproducibility.

## Main findings

- Claim occurrence is associated with driver, vehicle, occupation, coverage, and geographic-density variables.
- Logistic regression produced the best reported ROC AUC and remained directly interpretable.
- Older and retired policyholders were associated with lower predicted claim risk in the fitted logistic model.
- Male drivers, unemployed policyholders, type-E vehicles, and higher-density areas were associated with higher predicted risk.
- Moderate overall performance suggests that richer behavioural and claims-history features would be needed for production use.

## Responsible use

This is an educational project, not a production underwriting system. Variables such as gender and occupation may create legal, ethical, and fairness concerns. A real deployment would require bias testing, calibration, monitoring, governance, and review of applicable insurance regulation.

## Author

Yassine Zeamari — MSc in Data Science
