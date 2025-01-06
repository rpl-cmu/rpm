import tyro
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def plot_pix(pix_folder: str):
    pix_files = glob.glob(os.path.join(pix_folder, "*.npy"))
    if not pix_files:
        raise FileNotFoundError(f"No .npy files found in folder {pix_folder}")

    for pix_file in pix_files:
        print(f"Processing {pix_file}")
        data = np.load(pix_file)
        final = np.sum(data)

        fig = plt.figure(figsize=(10, 10))
        plt.xlim(-1e5, 1e5)
        plt.ylim(-1e5, 1e5)
        plt.scatter(np.real(data), np.imag(data), s=0.5)
        plt.plot([0, np.real(final)], [0, np.imag(final)], "r")
        plt.title(f"sum: {final}, abs: {np.abs(final)}")
        plt.savefig(pix_file.replace(".npy", ".png"), bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    cli = tyro.cli(plot_pix)
