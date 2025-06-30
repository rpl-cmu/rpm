import os
import tyro
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from os.path import join as pjoin

from utils import map_to_pts
from scipy.spatial import KDTree
import matplotlib.font_manager as fm
from matplotlib import rc

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    name="Times New Roman",
)
fm.fontManager.ttflist.insert(0, fe)
fontp = {"family": "Times New Roman", "size": 24, "weight": "bold"}
pdfp = {"fonttype": 42}

# rc('text', usetex=True)
rc("font", **fontp)
rc("pdf", **pdfp)


def chamfer_distance(pc1, pc2):
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)
    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)
    return np.concatenate([dist1, dist2])


def eval_pc(
    gt_file: str,
    sar_file: str,
    ra_file: str,
    cfar_file: str,
    output: str = "chamfer_distance_cdf.pdf",
) -> None:
    gt_map = pkl.load(open(gt_file, "rb"))
    eval_maps = {"Proposed": sar_file, "RA": ra_file, "CFAR": cfar_file}

    gt_pc = map_to_pts(gt_map["data"], gt_map["t"], gt_map["resolution"])
    cds = {}
    for key, file in eval_maps.items():
        eval_map = pkl.load(open(file, "rb"))
        eval_pc = map_to_pts(eval_map["data"], eval_map["t"], eval_map["resolution"])
        cds[key] = chamfer_distance(gt_pc, eval_pc)

    plt.figure()

    plt.xlabel("Chamfer Distance (m)")
    plt.ylabel("CDF")
    for key, cd in cds.items():
        sorted_cds = np.sort(cd)
        cdf = np.arange(1, len(sorted_cds) + 1) / len(sorted_cds)
        plt.plot(sorted_cds, cdf, label=key, linewidth=3)
    plt.legend()
    plt.grid(True)
    plt.savefig(output, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    tyro.cli(eval_pc)
