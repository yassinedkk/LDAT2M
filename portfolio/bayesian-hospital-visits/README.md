# Bayesian Hierarchical Modelling of Hospital Visits

An academic Bayesian-statistics project modelling patient hospital-visit counts while accounting for both patient characteristics and unobserved hospital-level heterogeneity.

**[Read the original coursework report](report.pdf)**

## Project overview

The response variable is the number of hospital visits observed for each patient over two years. Patients are nested within 30 hospitals, so a hierarchical count model is used to account for differences between hospitals.

The predictors are:

- patient age;
- chronic-disease status;
- hospital membership.

## Methodology

The project develops and evaluates hierarchical Poisson regression models using:

- hospital-specific random effects;
- Gamma and log-normal formulations for hospital heterogeneity;
- Gibbs sampling for conjugate hospital effects;
- Metropolis-Hastings updates for regression coefficients and hyperparameters;
- posterior summaries and 95% credible intervals;
- trace plots and posterior predictive checks;
- comparison between a manually implemented MCMC sampler and JAGS.

## Model

For patient $i$ in hospital $g(i)$:

```math
Y_i \mid \mu_i \sim \operatorname{Poisson}(\mu_i),
\qquad
\mu_i = v_{g(i)}\exp\!\left(\beta_0 + \beta_1\,\mathrm{age}_i + \beta_2\,\mathrm{chronic}_i\right).
```

The hospital effect $v_g$ captures unobserved variation between hospitals.

## Repository structure

```text
portfolio/bayesian-hospital-visits/
├── data/
│   ├── HospitalVisits.txt\n│   └── README.md
├── README.md
├── analysis.R
├── install_packages.R
└── report.pdf
```

- `report.pdf` is the original group report with student identification numbers removed for privacy.
- `analysis.R` preserves the original analysis. Only the dataset path and one obvious statement-order runtime error were corrected.
- `data/HospitalVisits.txt` contains 919 de-identified observations from 30 hospitals and is published with the user's explicit authorization.

## Reproduce the analysis

1. Install R, JAGS and the required R packages.
2. Run:

```r
source("install_packages.R")
```

3. Place an authorised copy of `HospitalVisits.txt` in `data/`.
4. Run:

```r
source("analysis.R")
```

## Skills demonstrated

Bayesian hierarchical modelling, count-data regression, MCMC, Gibbs sampling, Metropolis-Hastings, JAGS, posterior inference, posterior predictive checking, R and statistical interpretation.

## Authors

- Yassine Zeamari
- Louis Baltus

Academic project for LSTAT2130 - Introduction to Bayesian Statistics, UCLouvain, 2024-2025.
