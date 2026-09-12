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

    def check(self,pattern,tree_index,pattern_index,m):

        if (tree_index,pattern_index) in m:
            return m[(tree_index,pattern_index)]
        
        if  self.labels[tree_index] != pattern.labels[pattern_index]:

            m[(tree_index,pattern_index)]=False
            return False
        
        tree_child=self.children[tree_index]
        pattern_child=pattern.children[pattern_index]

        if len(pattern_child)==0:
            m[(tree_index,pattern_index)]=True
            return True
        
        position=0
        for child in pattern_child:
            ok=False
            for i in range(position,len(tree_child)):
                if self.check(pattern,tree_child[i],child,m):
                    ok=True
                    position=i+1
                    break
            if not ok:
                m[(tree_index,pattern_index)]=False
                return False
        m[(tree_index,pattern_index)]=True    
        return True
                
         
    def is_valid_subtree(self, pattern_tree):
        """
        Checks if the given pattern is an induced rooted subtree of this tree.
        To be implemented by the student.
        """
        root=pattern_tree.parents.index(-1)
        m={}
        for start in range(len(self.labels)):
            if self.labels[start] == pattern_tree.labels[root]:
             if self.check(pattern_tree,start,root,m):
                return True
        return False
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


def supprot(data,pattern):
    #count how many trees containt the pattern
    c=0
    for tree in data:
        if tree.is_valid_subtree(pattern):
            c+=1
    return c

def first_pattern_frequent(data,min_frequency):
    labels = set()
    # collect all label in dataset
    for tree in data:
        for label in tree.labels:
            labels.add(label)

    frequent=[]
    # check each single node pattern
    for label in labels:
        pattern=Tree([label],[-1])
        if supprot(data,pattern)/len(data)>=min_frequency:
            frequent.append(pattern)

    return frequent

def expande_pattern(pattern,labels):
    child=[]
    #get rightmost path of the tree
    path=Tree.get_rightmost_path(pattern.parents)
    #add a new node on each position of the path
    for p in path:
        for label in labels:
            new_labels=pattern.labels + [label]
            new_parents=pattern.parents + [p]

            child.append(Tree(new_labels,new_parents))
    return child

def dfs(data,pattern,min_frequency,labels,visited,output):

    k=(tuple(pattern.labels),tuple(pattern.parents))
    if k in visited:#avoid duplicates
        return
    visited.add(k)
    #count support
    if supprot(data,pattern)/len(data)< min_frequency:
        return
    
    output.append((pattern,supprot(data,pattern)/len(data)))
    #expend pattern rescursively
    for path in expande_pattern(pattern,labels):
        dfs(data,path,min_frequency,labels,visited,output)



def tree_to_sequence(labels, parents):
   
   #convert tree to sequence
    children = [[] for _ in range(len(labels))]
    for i, val in enumerate(parents):
        if val != -1:
            children[val].append(i)

    root = parents.index(-1)

    output = []
    stack = [(root, 0)] 

    while stack:
        node, child_index = stack[-1]

        if child_index == 0:
            output.append(labels[node])

        if child_index < len(children[node]):
            child = children[node][child_index]
            stack[-1] = (node, child_index + 1)
            stack.append((child, 0))
        else:
            stack.pop()
            if stack:
                output.append(-1)

    return output


def mine(filepath, min_frequency):
    """
    Main mining function to be implemented by the student.
    """
    dataset = Dataset(filepath)
    #get frequent pattern of size 1
    first_pattern=first_pattern_frequent(dataset,min_frequency)
    label=[]

    for pat in first_pattern:
        label.append(pat.labels[0])

    visited=set()
    output=[]
    #explore each pattern
    for pattern in first_pattern:
        dfs(dataset,pattern,min_frequency,label,visited,output)
    #print the result
    for pattern, frequent in output:
        print(Dataset.format_pattern(tree_to_sequence(pattern.labels, pattern.parents), frequent))

