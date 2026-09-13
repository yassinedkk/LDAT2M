# Dynamic Hedonic Price Indices - Kalman Filter vs DCS-t

Master's thesis project on the construction of **dynamic hedonic price
indices** using latent-factor models.

## Author

**Yassine Zeamari**  
Master in Data Science, Statistics orientation - UCLouvain

## Project overview

Prices of heterogeneous products cannot be compared directly because their
observable characteristics change. This project first uses a hedonic
regression to control for product characteristics, then estimates the remaining
common price movement as a time-varying latent factor.

Two methods are compared:

- **Kalman filter:** a classical Gaussian state-space estimator.
- **DCS-t model:** a Student-t score-driven estimator designed to reduce the
  influence of extreme observations.

The study combines Monte Carlo simulations, artificial crisis scenarios, and
an empirical application to 860 observations of French wines sold in Belgian
supermarkets.

## Key visual results

### Empirical dynamic price indices

![Comparison of empirical price indices](assets/empirical-price-indices.png)

### Robustness under artificial shocks

![Impact of artificial shocks on the latent factor](assets/crisis-robustness.png)

The empirical estimates are close under ordinary observations. Under injected
shocks, the DCS-t trajectory remains smoother because its adaptive weighting
reduces the influence of extreme residuals.

## Selected simulation results

### Random-walk state with Student-t observation errors

| N | nu | MSE Kalman | MSE DCS-t |
|---:|---:|---:|---:|
| 5 | 3 | 0.148 | 0.134 |
| 5 | 50 | 0.075 | 0.114 |
| 20 | 3 | 0.061 | 0.083 |
| 40 | 5 | 0.026 | 0.067 |
| 100 | 50 | 0.009 | 0.051 |

### DCS-t-generated latent state

| N | nu | MSE Kalman | MSE DCS-t |
|---:|---:|---:|---:|
| 5 | 3 | 0.01633 | 0.00152 |
| 20 | 3 | 0.01027 | 0.00031 |
| 40 | 3 | 0.01546 | 0.00015 |
| 60 | 3 | 0.00871 | 0.00009 |
| 100 | 50 | 0.00346 | 0.00005 |

### Crisis scenario

| N | nu | MSE Kalman | MSE DCS-t |
|---:|---:|---:|---:|
| 5 | 3 | 3.157 | 0.254 |
| 5 | 50 | 0.817 | 0.198 |
| 20 | 3 | 0.726 | 0.105 |
| 40 | 5 | 0.289 | 0.087 |
| 60 | 10 | 0.188 | 0.079 |
| 100 | 50 | 0.123 | 0.075 |

## Main conclusions

- The Kalman filter is generally more accurate under stable Gaussian-like
  dynamics.
- DCS-t is substantially more robust during crisis periods and in the presence
  of extreme observations.
- DCS-t clearly dominates when the latent state itself follows score-driven
  dynamics.
- Increasing the cross-sectional sample size improves latent-factor precision.
- In the empirical application, the high estimated degrees of freedom suggest
  residuals close to Gaussian behavior outside crisis periods.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab research_notebook.ipynb
```

The Monte Carlo sections are self-contained. The empirical section requires an
authorized copy of `wine.xlsx`; see [the data notice](DATA_NOTICE.md).

## Files

- [Research notebook](research_notebook.ipynb)
- [Master's thesis](master-thesis.pdf)
- [Defense slides](thesis-defense.pdf)
- `assets/`: selected figures for quick review
- `requirements.txt`: Python dependencies
- `DATA_NOTICE.md`: empirical-data availability and reproduction instructions

## Limitations and extensions

The empirical time dimension is short, some years contain few observations,
wine vintage is used as a proxy for time, and the specification has a single
latent factor. Possible extensions include joint hedonic-dynamic estimation,
multiple latent factors, asymmetric distributions, and regime-switching
models.

## Keywords

Kalman filter, DCS model, score-driven model, Student-t distribution, hedonic
price index, latent factor, Monte Carlo simulation, robust estimation, wine
prices.
