# Nonparametric Variance Estimation

This project studies variance-function estimation in a nonparametric regression model using Monte Carlo simulation. It compares two kernel-based approaches: a direct conditional-moment estimator inspired by Härdle and Vieu (1992), and a residual-based estimator inspired by Fan and Yao (1998).

## Problem setup

The simulations use the model

$$
Y_i = m(X_i) + 0.5\,\sigma(X_i)\varepsilon_i,
$$

with $X_i=i/n$, $\varepsilon_i\sim\mathcal{N}(0,1)$, $m(x)=\sin^3(2\pi x^3)$, and $\sigma^2(x)=2+\sin(2\pi x)$.

## Methods

- Gaussian-kernel regression;
- direct estimation through $E[Y^2\mid X=x]-m^2(x)$;
- residual-based variance estimation;
- 400-repetition Monte Carlo experiments;
- comparison through bias, variance, and mean squared error;
- sensitivity analysis for the mean and variance bandwidths;
- asymptotic experiments across several sample sizes.

## Main findings

- The residual estimator is generally more stable across the domain.
- The direct estimator performs competitively in some regions but becomes strongly biased near the variance minimum in the baseline experiment.
- Both methods are sensitive to bandwidth selection.
- Increasing the sample size reduces variance, while the remaining bias depends on the smoothing parameters.

## Repository structure

```text
portfolio/nonparametric-variance-estimation/
├── README.md
├── analysis.Rmd
└── report.pdf
```

The original course statement is not redistributed; only the student's analysis and report are included.

## Reproduce

Install R and the required plotting package:

```r
install.packages(c("rmarkdown", "ggplot2"))
```

Then render the analysis from RStudio or run:

```r
rmarkdown::render("analysis.Rmd")
```

## Author

Yassine Zeamari

Individual project for LSTAT2150, *Nonparametric Statistics (Smoothing Methods)*, UCLouvain (2024).
