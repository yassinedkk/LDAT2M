# Insurance Claim Prediction

An academic binary-classification project that predicts whether an automobile insurance policy will generate a claim during its coverage period.

**[Read the original coursework report](report.pdf)** — the complete report submitted for the LDATS2350 Data Mining course.

## Project overview

The dataset contains 24,774 one-year automobile insurance contracts. The target, `claimNumbMD`, indicates whether a claim occurred.

The project covers:

- exploratory data analysis;
- duplicate, missing-value and outlier checks;
- square-root transformation of vehicle value;
- standardisation and one-hot encoding;
- training and comparison of five classification models;
- interpretation of logistic-regression coefficients.

## Models and reported results

| Model | Accuracy | ROC AUC | Main observation |
|---|---:|---:|---|
| Logistic regression | 0.64 | 0.69 | Best balance between performance and interpretability |
| Decision tree | 0.64 | 0.56 | Interpretable, but weak discrimination |
| K-nearest neighbours | 0.61 | 0.64 | Best value selected: 19 neighbours |
| Multilayer perceptron | 0.61 | 0.68 | Similar discrimination, but less interpretable |
| Gaussian Naive Bayes | 0.61 | 0.67 | Highest recall for claims: 0.81 |

The original analysis recommends logistic regression because it combines balanced classification performance, the highest reported ROC AUC and direct interpretation of the predictors.

## Repository structure

```text
portfolio/insurance-claim-prediction/
├── data/
│   └── dataSetJune2025.csv
├── README.md
├── report.pdf
├── analysis.qmd
├── requirements.txt
└── .gitignore
```

- `report.pdf` is the original submitted report.
- `analysis.qmd` is Yassine Zeamari's original Quarto source. Only the local dataset path was made portable and the student identification number was removed for privacy.
- `data/dataSetJune2025.csv` contains the course dataset, published with the user's authorization.

## Reproduce the analysis

Requirements: Python 3.10+ and [Quarto](https://quarto.org/).

```bash
cd portfolio/insurance-claim-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
quarto render analysis.qmd
```

## Main conclusions

The available policyholder, vehicle and geographic variables provide only moderate predictive performance. Logistic regression remains the preferred model because it is competitive while allowing the direction and relative importance of associations to be discussed.

The original report identifies older age and retired status as associated with lower predicted risk, while male gender, unemployment, type-E vehicles and higher-density areas are associated with higher predicted risk. These are model associations, not causal conclusions.

## Limitations and responsible use

This is an educational project, not a production underwriting system. The results are based on one train/test split, and the available predictors do not capture richer information such as driving behaviour or complete claim history. Variables such as gender and occupation would require fairness, legal and governance review before operational use.

## Author

Yassine Zeamari — MSc in Data Science, Statistical Orientation, UCLouvain
