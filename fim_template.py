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


from itertools import combinations
from math import ceil

# -------------------------
# Apriori helpers
# -------------------------

def _hash_pair(i, j, B=200003):
    """Hash stable pour une paire (i<j)."""
    if i > j:
        i, j = j, i
    return (hash(i) * 1000003 + hash(j)) % B

def _gen_C2_hash(D_sorted, theta, B=200003):
    """
    Hash-based technique (cours) pour générer/pruner C2.
    D_sorted : liste de transactions TRIÉES (tuple d'int), déjà filtrées sur L1.
    """
    buckets = [0] * B

    # 1) scan: compter buckets
    for t in D_sorted:
        for i, j in combinations(t, 2):
            buckets[_hash_pair(i, j, B)] += 1

    # 2) candidats C2 uniquement si bucket >= theta
    C2 = set()
    for t in D_sorted:
        for i, j in combinations(t, 2):
            if buckets[_hash_pair(i, j, B)] >= theta:
                C2.add((i, j))

    return sorted(C2)

def _find_L1(D_sorted, theta):
    """Retourne L1 dict {(i,): count}."""
    counts = {}
    for t in D_sorted:
        for x in t:
            counts[x] = counts.get(x, 0) + 1
    L1 = {}
    for x, c in counts.items():
        if c >= theta:
            L1[(x,)] = c
    return L1

def _has_infrequent_subset(c, L_prev_set):
    """Prune Apriori: si un (k-1)-subset n'est pas fréquent, on rejette c."""
    k = len(c)
    for s in combinations(c, k - 1):
        if s not in L_prev_set:
            return True
    return False

def _apriori_gen(L_prev):
    """
    JOIN + PRUNE: génère Ck à partir de L_{k-1}.
    L_prev: liste de tuples triés (k-1 items).
    """
    L = sorted(L_prev)
    if not L:
        return []

    L_set = set(L)
    Ck = set()

    for i in range(len(L)):
        l1 = L[i]
        for j in range(i + 1, len(L)):
            l2 = L[j]

            # join condition: même préfixe de longueur k-2
            if len(l1) > 1 and l1[:-1] != l2[:-1]:
                break

            c = l1 + (l2[-1],)

            if not _has_infrequent_subset(c, L_set):
                Ck.add(c)

    return sorted(Ck)

def _subset_in_Ck(Ck_set, t_sorted, k):
    """Yield les k-subsets de t_sorted qui sont dans Ck_set."""
    for comb in combinations(t_sorted, k):
        if comb in Ck_set:
            yield comb


# -------------------------
# Required function (template)
# -------------------------

def mine_apriori(filepath, min_frequency):
    """
    Mine itemsets from filepath using Apriori.
    Print each frequent itemset on one line:
      [<item 1>, <item 2>, ...] (<frequency>)
    where frequency is RELATIVE support (float).
    """

    dataset = Dataset(filepath)
    n = len(dataset)
    if n == 0:
        return

    theta = ceil(min_frequency * n)  # support COUNT threshold

    # 1) Charger + trier chaque transaction UNE FOIS (gros gain)
    # Dataset donne des listes d'int ; on les transforme en tuples triés
    D_sorted = []
    for t in dataset:
        if t:
            D_sorted.append(tuple(sorted(t)))

    # 2) L1
    L1 = _find_L1(D_sorted, theta)
    L1_items = set(x[0] for x in L1.keys())

    # 3) Filtrer chaque transaction par L1 (réduction simple et sûre)
    D = []
    for t in D_sorted:
        ft = tuple(x for x in t if x in L1_items)
        if ft:
            D.append(ft)

    # Sortie: on doit imprimer tous les itemsets fréquents + fréquence relative
    # On imprime au fur et à mesure (pas besoin de stocker si tu veux)
    for it, cnt in sorted(L1.items()):
        freq = cnt / n
        print(f"[{', '.join(map(str, it))}] ({freq:.6f})")

    L_prev = sorted(L1.keys())  # itemsets fréquents taille 1
    k = 2

    # 4) Boucle niveaux k
    while L_prev:
        # --- Optimisation du cours: Hash-based pour k=2 ---
        if k == 2:
            Ck = _gen_C2_hash(D, theta)
        else:
            Ck = _apriori_gen(L_prev)

        if not Ck:
            break

        support = {c: 0 for c in Ck}
        Ck_set = set(Ck)

        # compter supports par scan DB
        for t in D:
            if len(t) < k:
                continue
            for c in _subset_in_Ck(Ck_set, t, k):
                support[c] += 1

        # construire Lk
        Lk = {it: cnt for it, cnt in support.items() if cnt >= theta}

        # imprimer Lk
        for it, cnt in sorted(Lk.items()):
            freq = cnt / n
            print(f"[{', '.join(map(str, it))}] ({freq:.6f})")

        L_prev = sorted(Lk.keys())
        k += 1
