from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from config.config import Config
from src.constants import DIM_FEATS, DIM_NUMBER_FEATS, LAYOUT_FEATS, NORMAL_FEATS, opcode_groups


def _read_npz(data: np.lib.npyio.NpzFile, exclude_keys: list[str]) -> dict[str, np.ndarray]:
    return {key: data[key] for key in data.files if key not in exclude_keys}


class GraphData(Data):
    pass


class Normalizer:
    def __init__(self, min_max_log: dict[int, tuple[float, float, bool]], normalize: bool) -> None:
        self.normalize = normalize
        idx = np.array(list(min_max_log.keys()), dtype=np.int64)
        min = np.array([val[0] for val in min_max_log.values()], dtype=np.float32)
        max = np.array([val[1] for val in min_max_log.values()], dtype=np.float32)
        is_log = np.array([val[2] for val in min_max_log.values()])
        self.idx = idx[~is_log]
        self.min = min[~is_log]
        self.max = max[~is_log]
        self.log_idx = idx[is_log]
        self.log_offset = 1 - min[is_log]
        self.log_max = np.log(max[is_log] + self.log_offset)

    def get(self, node_feat: np.ndarray) -> np.ndarray:
        values = node_feat[:, self.idx]
        log_values = np.log(node_feat[:, self.log_idx] + self.log_offset)
        if self.normalize:
            values = (values - self.min) / (self.max - self.min) * 2 - 1
        return np.concatenate([values, log_values], axis=1)


class FeatureReader:
    def __init__(self) -> None:
        self.dim_feat_normalizer = Normalizer(DIM_FEATS, True)
        self.dim_number_feat_normalizer = Normalizer(DIM_NUMBER_FEATS, False)
        self.layout_feat_normalizer = Normalizer(LAYOUT_FEATS, False)
        self.normal_feat_normalizer = Normalizer(NORMAL_FEATS, True)

    def read(self, node_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_node = len(node_feat)

        normal_feat = self.normal_feat_normalizer.get(node_feat)

        dim_feat = self.dim_feat_normalizer.get(node_feat).reshape(n_node, 6, -1)

        dim_number_feat = self.dim_number_feat_normalizer.get(node_feat).astype(np.int32)
        node_idx, dim_number_idx = np.where(dim_number_feat != 0)
        vals = dim_number_feat[dim_number_feat != 0]
        dim_number_rev_feat = np.zeros((len(node_feat), 6, dim_number_feat.shape[1]), dtype=node_feat.dtype)
        dim_number_rev_feat[node_idx, vals, dim_number_idx] = 1.0

        layout_feat = self.layout_feat_normalizer.get(node_feat)
        layout_feat[layout_feat == 0] = -1
        return normal_feat, np.concatenate([dim_feat, dim_number_rev_feat], -1), layout_feat


class KaggleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        npz_paths: list[str],
        sample_per_graph: int,
        cfg: Config,
        rng: np.random.Generator | None,
        config_from_npy: bool = True,
        n_config_lim: int = -1,
        dataset_type: str = "layout",
    ) -> None:
        super().__init__()
        exclude_keys = ["node_config_feat"] if config_from_npy else []
        self.gs = [_read_npz(np.load(fp), exclude_keys) for fp in npz_paths]
        if dataset_type == "tile":
            for g in self.gs:
                norm_coef = g["config_runtime_normalizers"] / g["config_runtime_normalizers"].mean()
                g["config_runtime"] = g["config_runtime"] / norm_coef
        self.sample_per_graph = sample_per_graph
        self.cfg = cfg
        self.dataset_type = dataset_type

        self.n_config = np.array([len(g["config_runtime"]) for g in self.gs], dtype=np.int64)
        if rng is None:
            self.config_perms = None
            self.n_sample = self.n_config
        else:
            self.n_sample = (np.sqrt(self.n_config * self.n_config.max(initial=-1)) * cfg.data_ratio).astype(int)
            self.config_perms = [rng.permutation(sample_each) for sample_each in self.n_sample]
        if n_config_lim != -1:
            self.n_sample = np.minimum(self.n_sample, n_config_lim)
        self.n_sample_acc = np.cumsum((np.insert(self.n_sample, 0, 0) - 1) // sample_per_graph + 1)
        self.config_from_npy = config_from_npy
        if config_from_npy:
            # **/layout/*/*/*/*.npz -> **/layout-config/*/*/*/*.npy  (Replace only the last "layout")
            self.node_config_feats = [
                np.load("layout-config".join(fp.rsplit("layout", 1))[:-3] + "npy", "r") for fp in npz_paths
            ]
        self.feature_reader = FeatureReader()

    def __len__(self) -> int:
        return self.n_sample_acc[-1]

    def _config_feat(self, graph_idx: int) -> np.ndarray:
        if self.config_from_npy:
            return self.node_config_feats[graph_idx]
        else:
            return self.gs[graph_idx]["node_config_feat"]

    def __getitem__(self, i: int) -> dict[str, Any]:
        for j, n_config in enumerate(self.n_sample_acc):
            if i < n_config:
                graph_idx = j - 1
                config_i = i - self.n_sample_acc[j - 1].item()
                config_idx = (
                    np.arange(config_i * self.sample_per_graph, (config_i + 1) * self.sample_per_graph)
                    % self.n_sample[graph_idx]
                )
                if self.config_perms is not None:
                    config_idx = self.config_perms[graph_idx][config_idx]
                config_idx %= self.n_config[graph_idx]
                break
        g = self.gs[graph_idx]
        num_nodes = len(g["node_opcode"])
        normal_node_feat, dim_feat, layout_feat_sub = self.feature_reader.read(g["node_feat"].copy())
        node_opcode = torch.from_numpy(g["node_opcode"]).to(torch.long)
        for opcode_group in self.cfg.opcode_groups:
            opcodes = opcode_groups[opcode_group]
            node_opcode[torch.isin(node_opcode, torch.tensor(opcodes))] = opcodes[0]
        # (1, n_config, 24)
        if self.dataset_type == "tile":
            tile_norm = np.log(g["config_feat"][config_idx] + 1)
            tile_feat = torch.from_numpy(np.stack([tile_norm[:, i : i + 6] for i in [0, 8, 16]], 1))[None]
        else:
            tile_feat = torch.zeros(1, self.sample_per_graph, 3, 6, dtype=torch.float32)

        if self.dataset_type == "layout":
            if self.cfg.layout_override:
                layout_feat = (
                    torch.from_numpy(layout_feat_sub)[:, None, None, :].expand(-1, self.sample_per_graph, 3, -1).clone()
                )
                layout_feat[g["node_config_ids"]] = (
                    torch.tensor(self._config_feat(graph_idx)[config_idx], dtype=torch.float32)
                    .transpose(0, 1)
                    .reshape(len(g["node_config_ids"]), self.sample_per_graph, 3, 6)
                )
            else:
                layout_feat = torch.full((num_nodes, self.sample_per_graph, 4, 6), -1.0)
                layout_feat[:, :, 0, :] = torch.from_numpy(layout_feat_sub)[:, None, :].expand(
                    -1, self.sample_per_graph, -1
                )
                layout_feat[g["node_config_ids"], :, 1:, :] = (
                    torch.tensor(self._config_feat(graph_idx)[config_idx], dtype=torch.float32)
                    .transpose(0, 1)
                    .reshape(len(g["node_config_ids"]), self.sample_per_graph, 3, 6)
                )
        else:
            layout_num = 4 - self.cfg.layout_override
            layout_feat = torch.full((num_nodes, self.sample_per_graph, layout_num, 6), -1.0)

        data = GraphData(
            x=torch.from_numpy(normal_node_feat),
            edge_index=torch.tensor(g["edge_index"].T, dtype=torch.long),
            y=torch.tensor(g["config_runtime"][config_idx] / 1e6, dtype=torch.float32)[None],
            node_opcode=node_opcode,
            dim_feat=torch.from_numpy(dim_feat),
            tile_feat=tile_feat,
            layout_feat=layout_feat,
        )

        # Add multiple virtual nodes
        if self.cfg.n_each_cluster != -1:
            n_cluster = (num_nodes - 1) // self.cfg.n_each_cluster + 1
            data.x = torch.cat([data.x, torch.zeros(n_cluster, data.x.shape[1])], dim=0)
            nums = np.full((n_cluster,), self.cfg.n_each_cluster)
            np.add.at(nums, np.random.choice(len(nums), nums.sum().item() - num_nodes, replace=True), -1)
            cluster_index = torch.repeat_interleave(torch.arange(len(nums)), torch.from_numpy(nums))
            data.edge_index = torch.cat(
                [
                    data.edge_index,
                    torch.stack([cluster_index + num_nodes, torch.arange(num_nodes)]),
                ],
                dim=1,
            )
            data.node_opcode = torch.cat([data.node_opcode, torch.zeros(n_cluster, dtype=torch.long)])
            data.dim_feat = torch.cat([data.dim_feat, torch.zeros(n_cluster, *data.dim_feat.shape[1:])])
            data.layout_feat = torch.cat([data.layout_feat, torch.full((n_cluster, *data.layout_feat.shape[1:]), -2.0)])
        return {
            "data": data,
            "graph_idx": graph_idx,
            "config_idx": torch.tensor(config_idx)[None],
            "is_tile": self.dataset_type == "tile",
        }
