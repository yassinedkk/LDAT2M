# Dynamic Hedonic Price Indices — Kalman Filter vs DCS-t

This project presents my master's thesis on the construction of **dynamic hedonic price indices** using latent factor models.

## Files

- [Master thesis](memoire.pdf)
- [Presentation slides](presentation.pdf)

## Project overview

The objective is to estimate a dynamic price index for heterogeneous goods. Prices are not directly comparable because products differ in observable characteristics. A hedonic model is first used to control for these characteristics, and the remaining component is modeled as a latent factor \(\beta_t\) evolving over time.

The project compares two estimation methods:

- **Kalman filter**: a classical Gaussian state-space approach.
- **DCS-t model**: a score-driven model based on the Student-t distribution, designed to be more robust to outliers and heavy-tailed errors.

## Why it matters

Dynamic hedonic indices are useful when product prices evolve over time while product characteristics also change. This type of modeling helps separate changes due to observable characteristics from a common latent market movement.

The comparison is important because real data can contain noise, extreme observations, and crisis periods. In such situations, a robust score-driven model may behave differently from a classical Gaussian filter.

## Methodology

The work combines:

1. **Monte Carlo simulations** to compare the models when the true latent factor is known.
2. **An empirical application** to French wines sold in Belgian supermarkets.
3. **Artificial crisis scenarios** to test robustness under extreme observations.

The models are evaluated using MSE, bias, and variance.

## Main results

- When the latent state follows a Gaussian random walk, the Kalman filter is generally more accurate.
- When the data contain crisis periods or extreme observations, the DCS-t model is more robust.
- When the true latent dynamics are score-driven, the DCS-t model clearly outperforms the Kalman filter.
- Increasing the number of observations improves the precision of latent factor estimation.
- In the empirical wine application, the estimated degrees of freedom parameter is high, suggesting that the residuals are close to Gaussian in the non-crisis case.

## Limitations and extensions

Main limitations include a short empirical time dimension, some years with few observations, the use of wine vintage as a proxy for time, and the use of a single latent factor.

Possible extensions include joint estimation of the hedonic and dynamic components, multiple latent factors, asymmetric distributions, and regime-switching models.

## Keywords

Kalman filter, DCS model, score-driven model, Student-t distribution, hedonic price index, latent factor, Monte Carlo simulation, robust estimation, wine prices.
