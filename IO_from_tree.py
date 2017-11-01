import numpy as np
import os
from scipy.linalg import block_diag


def IO_from_tree(tree, top=True):
    """Takes a tree of nested list form, where each element of the list is
    either another list or is a 1 to indicate a leaf. Produces a dataset
    corresponding to that tree"""
    if tree == 1:
	return [1]
    if tree == []:
	raise ValueError("lists cannot be empty! use 1 to indicate leaf nodes.")
    
    subtrees = [IO_from_tree(subtree, top=False) for subtree in tree]
    if top: # block_diag and add indicator to group these subtrees
	IO = np.concatenate(subtrees, axis=1)
    else:
	IO = block_diag(*subtrees)
	IO = np.concatenate((np.ones((len(IO), len(IO))), IO), axis=1)
    
    return IO 


if __name__ == "__main__":
    print IO_from_tree([[1,1],[1,1]])
    print IO_from_tree([
    [[1,1],[1,1,[1,1]]],
    [[1,1],[1,1,[1,1]]],
    ])



