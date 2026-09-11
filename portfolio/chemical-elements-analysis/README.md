# Chemical Elements: PCA and Discriminant Analysis

This project explores the physical and chemical properties of 50 common elements. It combines descriptive statistics, principal component analysis (PCA), linear discriminant analysis (LDA), and correspondence analysis to study relationships among quantitative properties and physical states.

## Data

The dataset contains the following variables:

- atomic mass;
- atomic radius;
- density;
- melting point;
- boiling point;
- molar heat capacity;
- physical state at room temperature.

The data source cited in the original project is [Éléments Chimiques](http://www.elementschimiques.fr/?fr). No explicit dataset license was supplied with the archive.

## Methods

- descriptive summaries by physical state;
- correlation analysis;
- PCA for dimensionality reduction and interpretation;
- supplementary qualitative-variable analysis;
- LDA to separate solid, liquid, and gaseous elements;
- correspondence analysis of categorized physical properties.

## Main findings

- The first two principal components explain approximately 78.1% of total variance.
- Density, boiling point, and melting point contribute strongly to the first component.
- Molar heat capacity contributes most strongly to the second component.
- Atomic mass, atomic radius, density, melting point, and boiling point are positively associated.
- Physical state helps explain part of the separation between elements, although the liquid class contains only one observation.

## Repository structure

```text
portfolio/chemical-elements-analysis/
├── README.md
├── analysis.Rmd
├── data/
│   └── elements.csv
└── report.pdf
```

## Reproduce

Install the required R packages:

```r
install.packages(c(
  "rmarkdown", "dplyr", "ggplot2", "cowplot",
  "corrplot", "FactoMineR", "factoextra", "MASS"
))
```

Then render the analysis:

```r
rmarkdown::render("analysis.Rmd")
```

## Author

Yassine Zeamari

Project for LSTAT2110, *Data Analysis*, UCLouvain (2025).
