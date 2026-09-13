# Dynamic Hedonic Price Indices: Kalman Filter vs DCS-t

This repository contains my master thesis and presentation on the construction of **dynamic hedonic price indices** using state-space and score-driven models.

## Documents

- [Master thesis: `memoire (26).pdf`](memoire%20%2826%29.pdf)
- [Presentation: `presentation (1).pdf`](presentation%20%281%29.pdf)

## Project overview

The objective of this work is to build a dynamic price index for heterogeneous goods. In many markets, observed prices are not directly comparable because products differ in their characteristics. For example, in the wine market, prices may depend on the region of production, wine type, quality rating, organic label, taste profile, alcohol level, acidity, sugar content, and other chemical characteristics.

A hedonic model is first used to control for observable characteristics. The remaining component is then modeled as a latent dynamic factor, denoted by \(\beta_t\), which represents the common evolution of prices over time after correcting for product characteristics.

The thesis compares two approaches for estimating this latent factor:

1. **Kalman filter**: a classical state-space method based on a Gaussian framework.
2. **DCS-t model**: a score-driven model based on the Student-t distribution, designed to be more robust to extreme observations and heavy-tailed errors.

## Why this topic matters

Dynamic hedonic price indices are useful when prices evolve over time and products are not perfectly comparable. They allow us to separate price changes due to product characteristics from changes linked to a common market component.

This is important in applied economics, finance, and data science because real-world data often contain:

- heterogeneous products;
- latent market dynamics;
- noisy observations;
- outliers or crisis periods;
- non-Gaussian behavior and heavy tails.

The comparison between the Kalman filter and DCS-t models helps identify when a classical Gaussian approach is sufficient and when a more robust score-driven approach becomes useful.

## Methodology

The work is divided into two main parts:

### 1. Monte Carlo simulations

Several simulation scenarios are considered:

- Gaussian random-walk latent state;
- Student-t observation errors;
- crisis periods with temporarily increased observation variance;
- latent state generated directly by a DCS-t dynamic.

For each configuration, 500 Monte Carlo simulations are performed. The models are evaluated using:

- Mean Squared Error (MSE);
- bias;
- variance.

### 2. Empirical application to wine prices

The models are applied to a dataset of **860 observations of French wines sold in Belgian supermarkets**. The dataset includes product characteristics such as region, wine type, sensory profile, organic label, quality rating, and physicochemical variables.

A hedonic regression is first estimated on log-prices. Then, the residual component is modeled dynamically using the Kalman filter and the DCS-t model.

## Main results

The simulations show that the relative performance of the two methods depends strongly on the data-generating process.

- When the latent state follows a Gaussian random walk, the Kalman filter is generally more accurate.
- When the data contain crisis periods or strong outliers, the DCS-t model is more robust.
- When the true latent dynamic is score-driven, the DCS-t model clearly outperforms the Kalman filter.
- Increasing the number of observations in the panel improves the precision of the latent factor estimation.

In the empirical wine application, the estimated degrees-of-freedom parameter is high, suggesting that the residuals are close to Gaussian in the non-crisis case. As a result, the Kalman filter and DCS-t model give relatively similar conclusions, although the DCS-t trajectory is smoother.

Artificial crisis scenarios are also introduced into the empirical data. In these cases, the Kalman filter reacts strongly to extreme observations, while the DCS-t model remains more stable because the Student-t score reduces the impact of large residuals.

## Limitations and extensions

The main limitations are:

- the empirical time dimension is short;
- some years contain few observations;
- the variable `year` is used as a proxy for time, although it corresponds to the wine vintage rather than the exact transaction date;
- the model uses a single latent factor;
- the DCS-t update may require normalization when the panel size becomes large.

Possible extensions include:

- estimating the hedonic and dynamic components jointly;
- introducing several latent factors;
- using asymmetric distributions;
- considering regime-switching models for crisis periods.

## Keywords

Kalman filter, DCS model, score-driven model, Student-t distribution, hedonic price index, latent factor, Monte Carlo simulation, robust estimation, wine prices.