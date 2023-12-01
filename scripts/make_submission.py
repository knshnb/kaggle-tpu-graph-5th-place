import argparse
import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training for Kaggle TPU Graph")
    parser.add_argument("--file_name", default="submission.csv")
    parser.add_argument("--layout_model_dirs", nargs="+", required=True)
    parser.add_argument("--tile_model_dirs", nargs="+", required=True)
    return parser.parse_args()


def get_layout_df(dirnames: list[str]) -> pd.DataFrame:
    ids, top_configs = [], []
    for model_type, search_type in itertools.product(["xla", "nlp"], ["default", "random"]):
        dir = f"input/npz_all/npz/layout/{model_type}/{search_type}/test"
        filenames = sorted(os.listdir(dir))
        data_list = [np.load(f"{dirname}/{model_type}-{search_type}-test.npz") for dirname in dirnames]
        for graph_idx in range(data_list[0]["graph_idx"].max() + 1):
            preds = []
            for data in data_list:
                graph_mask = data["graph_idx"] == graph_idx
                config_idx = data["config_idx"][graph_mask]
                assert (config_idx == np.arange(graph_mask.sum())).all()
                pred = data["pred"][graph_mask]
                preds.append((pred - pred.min()) / (pred.max() - pred.min()))
            ensembled_pred = np.mean(preds, axis=0)
            order = ensembled_pred.argsort()
            sample_name = filenames[graph_idx].split(".")[0]
            ids.append(f"layout:{model_type}:{search_type}:{sample_name}")
            top_configs.append(";".join(map(str, order.tolist())))
    return pd.DataFrame({"ID": ids, "TopConfigs": top_configs})


def get_tile_df(dirnames: list[str]) -> pd.DataFrame:
    ids, top_configs = [], []
    filenames = sorted(os.listdir(f"input/npz_all/npz/tile/xla/test"))
    data_list = [np.load(f"{dirname}/tile-test.npz") for dirname in dirnames]
    for graph_idx in tqdm(range(data_list[0]["graph_idx"].max() + 1)):
        preds = []
        for data in data_list:
            graph_mask = data["graph_idx"] == graph_idx
            config_idx = data["config_idx"][graph_mask]
            assert (config_idx == np.arange(graph_mask.sum())).all()
            pred = data["pred"][graph_mask]
            preds.append((pred - pred.min()) / (pred.max() - pred.min()))
        ensembled_pred = np.mean(preds, axis=0)
        order = ensembled_pred.argsort()
        sample_name = filenames[graph_idx].split(".")[0]
        ids.append(f"tile:xla:{sample_name}")
        top_configs.append(";".join(map(str, order[:5].tolist())))
    return pd.DataFrame({"ID": ids, "TopConfigs": top_configs})


def main(args: argparse.Namespace) -> None:
    layout_df = get_layout_df(args.layout_model_dirs)
    tile_df = get_tile_df(args.tile_model_dirs)
    pd.concat([layout_df, tile_df]).to_csv(args.file_name, index=False)


if __name__ == "__main__":
    main(parse())
