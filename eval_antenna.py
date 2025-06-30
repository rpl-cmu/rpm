import os
import tyro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
import matplotlib.font_manager as fm
from matplotlib import rc

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    name="Times New Roman",
)
fm.fontManager.ttflist.insert(0, fe)
fontp = {"family": "Times New Roman", "size": 18, "weight": "bold"}
pdfp = {"fonttype": 42}

# rc('text', usetex=True)
rc("font", **fontp)
rc("pdf", **pdfp)


def read_single(folder: str, name: str):

    file = pjoin(folder, name, "eval.txt")
    df = pd.read_csv(file, sep=", ")
    return df


def main(folder: str):
    files = os.listdir(folder)
    files.sort()

    seqs = pd.DataFrame(columns=["name", "n_device", "CD", "HD", "F-score"])
    for f in files:
        name = "_".join(f.split("_")[:-1])
        ndevice = f.split("_")[-1]
        df = read_single(folder, f)
        df['name'] = name
        df['n_device'] = ndevice
        seqs = pd.concat([seqs, df], ignore_index=True)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    sns.lineplot(data=seqs, x="n_device", y="CD", hue="name", marker='o', markersize=6, markerfacecolor='.5', ax=axes[0], legend=False)

    sns.lineplot(data=seqs, x="n_device", y="HD", hue="name", marker='o', markersize=6, markerfacecolor='.5', ax=axes[1], legend=False)

    sns.lineplot(data=seqs, x="n_device", y="F-score", hue="name", marker='o', markersize=6, markerfacecolor='.5', ax=axes[2])
    axes[2].get_legend().remove()

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center', ncol=10)


    plt.savefig("ndevice.pdf", bbox_inches='tight', format='pdf')

if __name__ == "__main__":
    tyro.cli(main)
