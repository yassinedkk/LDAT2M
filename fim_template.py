"""
This file given to you as a skeleton for your implementations of frequent itemsets mininh algorithms.
You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori, eclat and fpgrowth methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
"""

import re
from pathlib import Path

PATTERN_RE = re.compile(r"\[((?:\d+,? ?)+)\] *\(\d+\.\d+\)")

class Dataset:
    """Utility class to manage a dataset stored in a external file.
    You can modfy this however you want."""

    def __init__(self, path):
        self.path = Path(path)
        self.items = set()
        self.transactions = []

        self._load_data()

    def _load_data(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.path}")

        with self.path.open('r', encoding='utf-8') as file:
            for line in file:
                transaction = list(map(int, line.strip().split()))
                if not transaction:
                    continue
                self.transactions.append(transaction)
                self.items.update(transaction)

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, index):
        return self.transactions[index]

    def __iter__(self):
        return iter(self.transactions)

    def __repr__(self):
        return (f"Dataset(name='{self.path.name}', "
                f"transactions={len(self)}, "
                f"unique_items={len(self.items)})")

    @property
    def num_items(self) -> int:
        return len(self.items)



def get_patterns_from_file(filename):
    """Parse itemset patterns from a file, returning None if any lines are malformed."""
    patterns = set()
    errors = []

    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            match = PATTERN_RE.search(line)
            if match is None:
                errors.append(line)
            else:
                itemset = tuple(sorted(int(x) for x in match.group(1).split(', ')))
                patterns.add(itemset)

    if errors:
        print(f"[ERROR] {len(errors)} malformed line(s) in '{filename}':")
        for line in errors:
            print(f"\t{line}")
        return None

    return patterns


def _show_diff(label, patterns, limit = 10):
    """Print a sample of patterns from a diff set."""
    to_show = list(patterns)[:limit]
    print(f"{label}:")
    for pattern in to_show:
        print(f"\t{pattern}")
    print(f"(Showing {len(to_show)} of {len(patterns)})")


def compare_solution_files(expected_file, actual_file):
    """Compare patterns in actual_file against expected_file, reporting mismatches."""
    expected = get_patterns_from_file(expected_file)
    actual = get_patterns_from_file(actual_file)

    if expected is None or actual is None:
        return

    missed = expected - actual
    excess = actual - expected

    if not missed and not excess:
        print("The files contain the same patterns.")
        return

    if missed:
        _show_diff("Missed itemsets from expected file", missed)
    if excess:
        _show_diff("Unexpected itemsets not in expected file", excess)


def mine_apriori(filepath, min_frequency):
    """Mine itemsets from a filepath using apriori. We recommend having a file per algorithm"""
    pass