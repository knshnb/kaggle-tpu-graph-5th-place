import argparse
import glob
import os

import numpy as np
from tqdm import tqdm


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training for Kaggle TPU Graph")
    parser.add_argument("--in_base_dir", default="input")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    fps = glob.glob(f"{args.in_base_dir}/npz_all/npz/layout/*/*/*/*.npz")
    for fp in tqdm(fps):
        node_config_feat = np.load(fp)["node_config_feat"].astype(np.int8)
        tmp_names = fp.split("/")
        tmp_names[-5] = "layout-config"
        tmp_names[-1] = tmp_names[-1].rsplit(".", 1)[0] + ".npy"
        out_fp = "/".join(tmp_names)
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        np.save(out_fp, node_config_feat)


if __name__ == "__main__":
    main(parse())
