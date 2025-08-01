import numpy as np
from scipy.spatial import KDTree
import point_cloud_utils as pcu


def chamfer_distance(gt_pc, eval_pc):
    cd = pcu.chamfer_distance(gt_pc, eval_pc)
    return cd


def hausdorff_distance(gt_pc, eval_pc):
    hd = pcu.hausdorff_distance(gt_pc, eval_pc)
    return hd


def f_score(gt_pc, eval_pc, thresh_dist=0.2, eps=1e-8):

    tree_gt = KDTree(gt_pc)
    tree_eval = KDTree(eval_pc)
    distances_gt_to_eval = tree_eval.query(gt_pc)[0]
    distances_eval_to_gt = tree_gt.query(eval_pc)[0]

    precision = np.mean(distances_eval_to_gt < thresh_dist)
    recall = np.mean(distances_gt_to_eval < thresh_dist)

    # Compute F-score
    f_score = 2 * (precision * recall) / (precision + recall + eps)
    return f_score
