# Frequent Itemset Mining: Apriori, Eclat, and FP-Growth

Portfolio project for **LINGI2364** comparing three frequent itemset mining algorithms on transactional data.

## Project overview

The project implements Apriori with hash-based pruning and transaction filtering, Eclat with a vertical transaction-ID representation, and FP-Growth with recursive conditional FP-trees. Each implementation accepts a dataset path and a minimum frequency. Input files contain one transaction per line, with integer item identifiers separated by spaces.

## Results

The full methodology, figures, and benchmark tables are available in the [cleaned project report](report.pdf).

### Retail dataset

The three algorithms showed similar execution times because the transactions contain relatively few items and generate few frequent patterns. Eclat was slightly faster, while Apriori and FP-Growth used less memory.

| Minimum support | Apriori memory | Eclat memory | FP-Growth memory |
| ---: | ---: | ---: | ---: |
| 0.9 | 45.15 MB | 86.46 MB | 45.15 MB |
| 0.5 | 48.57 MB | 86.46 MB | 45.15 MB |
| 0.1 | 50.23 MB | 90.58 MB | 45.15 MB |

### Chess dataset

The dense chess dataset produced a rapid increase in frequent patterns as minimum support decreased. Apriori became substantially slower. At support 0.8, its reported runtime exceeded 700 seconds, while Eclat and FP-Growth required only a few seconds.

| Minimum support | Eclat time | Eclat memory | FP-Growth time | FP-Growth memory |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.9980 s | 15.68 MB | 1.5938 s | 2.50 MB |
| 0.7 | 22.4600 s | 37.70 MB | 20.3200 s | 19.76 MB |
| 0.5 | 571.8755 s | 345.28 MB | 290.7852 s | 354.19 MB |
| 0.4 | 3,315.6225 s | 1,882.08 MB | 1,491.6210 s | 1,910.65 MB |

### Accidents dataset

On the large but less dense accidents dataset, Eclat was generally faster but consumed much more memory. FP-Growth offered the strongest time-memory compromise in the reported experiments.

| Minimum support | Eclat time | Eclat memory | FP-Growth time | FP-Growth memory |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 68.8717 s | 782.23 MB | 125.5266 s | 230.76 MB |
| 0.7 | 106.5313 s | 1,282.28 MB | 147.8698 s | 232.80 MB |
| 0.5 | 479.9387 s | 2,308.29 MB | 446.8456 s | 283.41 MB |
| 0.4 | 1,248.1719 s | 2,816.57 MB | 2,408.0359 s | 395.24 MB |

Overall, Apriori used relatively little memory but scaled poorly on dense data. Eclat was often fast but memory intensive. FP-Growth generally provided the best compromise. These values are the experimental results reported in the submitted report and were not rerun during portfolio packaging.

## Usage

```python
from apriori import mine_apriori
from eclat import mine_eclat
from fp_growth import mine_fpgrowth

mine_apriori("dataset.dat", 0.5)
mine_eclat("dataset.dat", 0.5)
mine_fpgrowth("dataset.dat", 0.5)
```

## Files

- `apriori.py`: level-wise candidate generation and pruning
- `eclat.py`: vertical-format depth-first mining
- `fp_growth.py`: FP-tree construction and recursive mining
- `report.pdf`: anonymized report with benchmarks and figures

## Author

Yassine Zeamari
