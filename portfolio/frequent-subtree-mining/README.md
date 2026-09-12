# Frequent Subtree Mining

Portfolio project for **LINGI2364 - Mining Patterns in Data**. The project
adapts a pattern-growth approach inspired by PrefixSpan to mine frequent
induced rooted subtrees from databases of rooted ordered trees.

## Project overview

Trees are encoded as pre-order sequences. Integer labels represent nodes and
`-1` marks a return to the parent node. The implementation:

- parses each sequence into node-label and parent-index arrays;
- checks whether a pattern is an induced rooted subtree;
- grows patterns along their rightmost path;
- explores candidates with depth-first search;
- prunes infrequent and previously visited patterns.

## Usage

Run the miner from Python by passing a dataset path and a minimum relative
support:

```python
from connected_subtrees import mine

mine("datasets/toy.trees", 0.5)
mine("datasets/small_1.trees", 0.1)
```

Each frequent pattern is printed as a `-1`-delimited pre-order sequence,
followed by its relative support.

## Results

The experiments reported in `report.pdf` evaluate runtime, memory use, and the
number of frequent patterns across several support thresholds. On the toy
dataset, reducing support from `0.90` to `0.10` increased the number of patterns
from 1 to 25. The smaller benchmark remained inexpensive because it generated
few frequent patterns.

## Project structure

- `connected_subtrees.py`: frequent induced subtree mining implementation
- `datasets/toy.trees`: toy tree database
- `datasets/small_1.trees`: small tree database
- `instructions/Instruction.pdf`: original project specification
- `instructions/Instruction.tex`: LaTeX source of the specification
- `solutions/`: reference outputs supplied with the assignment
- `sim_template.py`: original starter template
- `report.pdf`: anonymized project report

## Author

Yassine Zeamari
