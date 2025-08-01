import argparse
from os.path import join as pjoin
from mmwcas.dataset import MIMODataset


argparser = argparse.ArgumentParser(description="load MIMO dataset")
argparser.add_argument("--folder", type=str, help="Folder path", required=True)
argparser.add_argument(
    "--radar", type=str, help="Radar name (radar0, radar1)", default="radar0"
)
args = argparser.parse_args()

dataset = MIMODataset(pjoin(args.folder, args.radar))

print(dataset.adc.param)

for sig, pose_tx, pose_rx, stamp in dataset:
    print(sig.shape, pose_tx.shape, pose_rx.shape, stamp)
    break
