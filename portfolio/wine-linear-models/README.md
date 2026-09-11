# Wine Citric Acid Prediction with Linear Models

This project develops and evaluates a linear regression model for predicting citric acid concentration from the physicochemical properties of red wine samples.

## Data

The dataset contains 599 observations and 12 variables, including:

- fixed and volatile acidity;
- citric acid;
- residual sugar and chlorides;
- free and total sulfur dioxide;
- density and pH;
- sulphates, alcohol, and quality.

The supplied data file is semicolon-delimited and is stored as `data/wine_data.csv`. The original archive did not include an explicit dataset license.

## Methods

- 90/10 training-validation split;
- descriptive statistics, boxplots, and correlation analysis;
- multiple linear regression with quantitative and categorized predictors;
- individual and partial F-tests;
- model selection using adjusted R-squared, AIC, BIC, and prediction error;
- residual, leverage, and influence diagnostics;
- tests for multicollinearity, heteroskedasticity, and autocorrelation;
- remedial modeling with weighted least squares and the Cochrane-Orcutt procedure;
- 95% prediction intervals on the validation set.

## Main findings

- Volatile acidity has a strong negative association with citric acid.
- Fixed acidity and chlorides have positive associations with citric acid.
- Alcohol categories and wine quality add statistically significant information.
- The autocorrelation detected in the initial model was addressed with a Cochrane-Orcutt transformation.
- Prediction-interval coverage on the validation set was 90%, below the nominal 95% level.
- The model remains useful for interpreting relationships, but its validation performance suggests that nonlinear alternatives could improve prediction.

## Repository structure

```text
portfolio/wine-linear-models/
├── README.md
├── data/
│   └── wine_data.csv
└── report.pdf
```

The original course statement is not redistributed. The archive did not contain the editable R source; the submitted report includes the relevant code and results.

## Authors

- Mateus Auza Cruz
- Yassine Zeamari

Group project for LSTAT2120, *Linear Models*, UCLouvain (2024-2025).
