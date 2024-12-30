import numpy as np
from scipy.spatial import KDTree

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    print(A.shape, B.shape)
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)