import os
import tyro
import numpy as np
import open3d as o3d
import pickle as pkl
from os.path import join as pjoin

from evaluation import metric
from utils import map_to_pts


def eval_pc(seq_dir: str):
    gt_file = pjoin(seq_dir, "map/lidar.pkl")
    eval_file = pjoin(seq_dir, "map/cfar.pkl")
    gt_map = pkl.load(open(gt_file, "rb"))
    eval_map = pkl.load(open(eval_file, "rb"))

    gt_pc = map_to_pts(gt_map["data"], gt_map["t"], gt_map["resolution"])
    eval_pc = map_to_pts(eval_map["data"], eval_map["t"], eval_map["resolution"])

    cd = metric.chamfer_distance(gt_pc, eval_pc)
    print(f"Chamfer Distance: {cd}")

    hd = metric.hausdorff_distance(gt_pc, eval_pc)
    print(f"Hausdorff Distance: {hd}")

    f_score = metric.f_score(gt_pc, eval_pc)
    print(f"F-score: {f_score}")

    gt_o3d_pc = o3d.geometry.PointCloud()
    gt_o3d_pc.points = o3d.utility.Vector3dVector(gt_pc)
    gt_o3d_pc.paint_uniform_color([1, 0, 0])  # Red for ground truth

    eval_o3d_pc = o3d.geometry.PointCloud()
    eval_o3d_pc.points = o3d.utility.Vector3dVector(eval_pc)
    eval_o3d_pc.paint_uniform_color([0, 1, 0])  # Green for evaluation

    o3d.visualization.draw_geometries(
        [gt_o3d_pc, eval_o3d_pc], window_name="Point Cloud Visualization"
    )


if __name__ == "__main__":
    tyro.cli(eval_pc)
