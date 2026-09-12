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

from collections import defaultdict
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

def node(item,frequency,parent):
    # create node of the FP-tree
    return{"item": item, "frequency": frequency, "parent": parent, "children": {}, "next": None}
   
from collections import defaultdict

def fp_tree(D, mins):
    #count global item support
    counts = defaultdict(int)

    
    for t in D:
        # weighted transactions for conditional tree
        if (
            isinstance(t, tuple)
            and len(t) == 2
            and isinstance(t[0], (list, tuple))
            and isinstance(t[1], int)
        ):
            items, c = t
        else:
            items, c = t, 1

        for i in items:
            counts[i] += c

    # keep only items that are frequent
    counts = {item: c for item, c in counts.items() if c >= mins}
    if not counts:
        return None, None

    #create root of the FP-tree and header table 
    root = node(None, 0, None)
    table = {item: [c, None] for item, c in counts.items()}

    # insert transactions into the FP-tree
    for t in D:
        if (
            isinstance(t, tuple)
            and len(t) == 2
            and isinstance(t[0], (list, tuple))
            and isinstance(t[1], int)
        ):
            items, c = t
        else:
            items, c = t, 1
#remove infrequent items and sort by frequency
        filtered = [x for x in items if x in counts]
        if not filtered:
            continue

        filtered.sort(key=lambda x: (-counts[x], x))
        current = root

        for item in filtered:
            child = current["children"].get(item)
# if node exist increase count
            if child is not None:
                child["frequency"] += c

            else:
                # create new node
                child = node(item, c, current)
                current["children"][item] = child
                #update header table
                if table[item][1] is None:
                    table[item][1] = child
                else:
                    n = table[item][1]
                    while n["next"] is not None:
                        n = n["next"]
                    n["next"] = child

            current = child

    return root, table



def fp_growth_mining(table,  prefix, min_support):
    
    global frequent

    def sort(t):
        item,d=t
        return (d[0], item)
    
    items=sorted(table.items(), key=sort)

    for p in items:
        item=p[0]
        support=p[1][0]
        node_f= p[1][1]
    # create new pattern by adding item to prefix
        new_pattern =tuple( sorted(prefix + [item]))
        frequent[new_pattern]=support
# build conditional pattern base 
        cond_pattern = []
        node= node_f
        
        while node is not None:
            path = []
            parent = node["parent"]
            
             # move up in the tree to get the prefix path   
            while parent is not None and parent["item"] is not None:
                    path.append(( parent["item"]))
                    parent = parent["parent"]                                                                                             
            if path:    
                    path.reverse()
                    cond_pattern.append((path, node["frequency"]))
            node = node["next"]
            # build conditional FP-tree 
        _, table_cond = fp_tree(cond_pattern, min_support)
        # recursively mine conditional FP-tree
        if table_cond:
            fp_growth_mining(table_cond, list(new_pattern), min_support)

from math import ceil

def mine_fpgrowth(filepath, min_frequency):
    global frequent
    frequent = {}
    dataset = Dataset(filepath)
    n = len(dataset)
    threshold = ceil(min_frequency * n)
    # sort transactions 
    D = [tuple(sorted(t)) for t in dataset]
    # build initial FP-tree
    root, table = fp_tree(D, threshold)
    if table is None:
        return
   # start recursive mining 
    fp_growth_mining(table, [],threshold)

    for item, support in sorted(frequent.items()):
        frq = support / n
        print(f"[{', '.join(map(str, item))}] ({frq:.6f})")
