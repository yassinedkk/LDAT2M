from collections import defaultdict


class Tree:
    """
    Represents a rooted ordered tree using arrays for labels and parent indices.
    
    Attributes:
        labels (list[int]): The label of each node in pre-order.
        parents (list[int]): The index of the parent for each node. The root has parent -1.
        size (int): Total number of nodes in the tree.
        children (list[list[int]]): Adjacency list mapping each node index to its children's indices.
        label_index (dict[int, list[int]]): Maps each label to the list of node indices carrying that label.
    """
    def __init__(self, labels, parents):
        self.labels = labels    
        self.parents = parents 
        self.size = len(labels)
        
        # Build adjacency list for children
        self.children = [[] for _ in range(self.size)]
        for i, p in enumerate(parents):
            if p != -1:
                self.children[p].append(i)
        
        # Build label index for fast lookup of starting points
        self.label_index = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_index[label].append(i)
        
    def is_valid_subtree(self, pattern_tree):
        """
        Checks if the given pattern is an induced rooted subtree of this tree.
        To be implemented by the student.
        """
        raise NotImplementedError("This method must be implemented for Task 1.")
    
    def __repr__(self):
        """Returns a nested parenthesis representation of the tree."""
        def get_repr(idx):
            child_strs = [get_repr(c) for c in self.children[idx]]
            if not child_strs:
                return str(self.labels[idx])
            return f"({self.labels[idx]} {' '.join(child_strs)})"
        
        if not self.labels:
            return ""
        root_idx = self.parents.index(-1) if -1 in self.parents else 0
        return get_repr(root_idx)
    
    @staticmethod
    def get_rightmost_path(parents):
        """
        Returns the indices of the nodes on the rightmost path of a tree 
        defined by its parents list.
        """
        if not parents:
            return []
        path = []
        idx = len(parents) - 1
        while idx != -1:
            path.append(idx)    
            idx = parents[idx]
        return path[::-1]


class Dataset:
    """
    A collection of trees loaded from a database file.
    """
    def __init__(self, path):
        self.trees = []
        self._load_trees(path)
    
    def _load_trees(self, path):
        """Parses the input file where each line is a -1 delimited tree sequence."""
        with open(path, 'r') as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                tree = Dataset.parse_to_tree(tokens)
                if tree:
                    self.trees.append(tree)
    
    @staticmethod
    def parse_to_tree(tokens):
        """
        Converts a -1 delimited sequence of labels into a Tree object.
        Example: [1, 2, -1, 3] -> Tree(labels=[1, 2, 3], parents=[-1, 0, 0])
        """
        if not tokens:
            return None
            
        labels, parents, stack = [], [], []
        for val in tokens:
            if val == -1:
                if stack:
                    stack.pop()
            else:
                curr_idx = len(labels)
                parent_idx = stack[-1] if stack else -1
                labels.append(val)
                parents.append(parent_idx)
                stack.append(curr_idx)
        
        return Tree(labels, parents)

    @staticmethod
    def format_pattern(tokens, support):
        """
        Formats a pattern and its support for printing.
        Args:
            tokens: list of labels in -1 delimited format.
            support: float (relative frequency between 0.0 and 1.0).
        """
        t_list = list(tokens)
        # Remove trailing -1s for cleaner output
        while t_list and t_list[-1] == -1:
            t_list.pop()
        
        return f"[{', '.join(map(str, t_list))}] ({support:.6f})"
    
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return self.trees[idx]

    def __iter__(self):
        return iter(self.trees)


def mine(filepath, min_frequency):
    """
    Main mining function to be implemented by the student.
    """
    dataset = Dataset(filepath)
    # TODO: Implement your tree mining algorithm here
    pass
