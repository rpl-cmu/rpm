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

        cov = np.cov(np.real(data), np.imag(data))
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = plt.matplotlib.patches.Ellipse(
            xy=(np.mean(np.real(data)), np.mean(np.imag(data))),
            width=lambda_[0] * 2,
            height=lambda_[1] * 2,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            edgecolor="b",
            fc="None",
            lw=2,
        )
        plt.gca().add_patch(ell)

        cov = np.array([[1000., 0.], [0., 1000.]])
        mean = np.array([0., 0.])
        for d in data:
            d = np.array([np.real(d), np.imag(d)])
            u = np.linalg.norm(d)
            # print(u)
            m_cov = np.diag([u, u])

            k = cov @ np.linalg.inv(cov + m_cov)
            mean += k @ (d - mean)
            cov = cov - k @ cov
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell2 = plt.matplotlib.patches.Ellipse(
            xy=(np.mean(np.real(data)), np.mean(np.imag(data))),
            width=lambda_[0] * 2,
            height=lambda_[1] * 2,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            edgecolor="y",
            fc="None",
            lw=2,
        )
        plt.gca().add_patch(ell2)
        mean *= len(data)
        plt.plot([0, mean[0]], [0, mean[1]], "g")
        print(mean, cov)

        plt.title(f"sum: {final}, abs: {np.abs(final)}")
        plt.savefig(pix_file.replace(".npy", ".png"), bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    cli = tyro.cli(plot_pix)
