import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import dropout_edge

from config.config import Config
from src.dataset import GraphData


def get_graph_conv(conv_type: str, in_ch: int, out_ch: int):
    if conv_type == "sage":
        return SAGEConv(in_ch, out_ch)
    elif conv_type == "gcn":
        return GCNConv(in_ch, out_ch)
    elif conv_type == "gat":
        return GATConv(in_ch, out_ch)
    elif conv_type == "gatv2":
        return GATv2Conv(in_ch, out_ch)
    elif conv_type == "gin":
        return GINConv(torch.nn.Sequential(torch.nn.Linear(in_ch, out_ch), torch.nn.ReLU()))
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class GNN(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.opcode_embed = torch.nn.Embedding(121, cfg.opcode_embed_dim)
        in_ch = 53 + cfg.opcode_embed_dim + 6 * 23 + 6 * 3 + 6 * (4 - cfg.layout_override)
        if cfg.use_pre_linear:
            self.pre_linear = torch.nn.Sequential(torch.nn.Linear(in_ch, cfg.mid_ch), torch.nn.ReLU())
        self.convs = torch.nn.ModuleList()
        self.rev_convs = torch.nn.ModuleList()
        for i in range(cfg.n_layer):
            tmp_in = cfg.mid_ch if i > 0 or cfg.use_pre_linear else in_ch
            self.convs.append(get_graph_conv(cfg.conv_type, tmp_in, cfg.mid_ch // 2))
            self.rev_convs.append(get_graph_conv(cfg.conv_type, tmp_in, cfg.mid_ch // 2))
        self.head = torch.nn.Linear(cfg.mid_ch * cfg.concat_last_n, 1)
        self.is_static_graph = "gat" not in cfg.conv_type

    def forward(self, data: Batch) -> torch.Tensor:
        # (batch_size, num_nodes * batch_size, in_ch) -> (batch_size, sample_per_graph)
        batch_size = data.batch_size
        opcode_embed = self.opcode_embed(data.node_opcode)
        sample_per_graph = data.layout_feat.shape[1]
        x = torch.cat(
            [
                data.x[None].expand(sample_per_graph, -1, -1),
                opcode_embed[None].expand(sample_per_graph, -1, -1),
                data.dim_feat.flatten(1, 2)[None].expand(sample_per_graph, -1, -1),
                data.layout_feat.transpose(0, 1).flatten(2, 3),
                data.tile_feat[data.batch].transpose(0, 1).flatten(2, 3),
            ],
            dim=-1,
        )  # (sample_per_graph, num_nodes * batch_size, 140 + opcode_embed_dim + 18)
        if self.cfg.use_pre_linear:
            x = self.pre_linear(x)
        if not self.is_static_graph:
            data.x = x.transpose(0, 1)
            data_list = [
                GraphData(x=x_each, edge_index=g.edge_index, batch=g.batch)
                for g in data.to_data_list()
                for x_each in g.x.transpose(0, 1)
            ]
            data = Batch.from_data_list(data_list)
            x = data.x
        xs = []
        drop_edge_ratio = random.uniform(0.0, self.cfg.drop_edge_ratio_max)
        edge_index = dropout_edge(data.edge_index, p=drop_edge_ratio, training=self.training)[0]
        for i in range(len(self.convs)):
            x = torch.cat([self.convs[i](x, edge_index), self.rev_convs[i](x, torch.flip(edge_index, (0,)))], dim=-1)
            x = F.relu(x)
            if i >= len(self.convs) - self.cfg.concat_last_n:
                xs.append(x)
        x = torch.cat(xs, dim=-1)
        if not self.is_static_graph:
            return self.head(global_add_pool(x, data.batch)).reshape(batch_size, -1, 2)
        else:
            return self.head(global_add_pool(x, data.batch)).transpose(0, 1)


class Transformer(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, n_head: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU())
        encoder_layer = torch.nn.TransformerEncoderLayer(hidden_dim, n_head, hidden_dim, dropout=0.0, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, in_dim) -> (B, hidden_dim)"""
        x = self.mlp(x)
        return self.transformer(x)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


class DimensionAttentionGNN(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.opcode_embed = torch.nn.Embedding(121, cfg.opcode_embed_dim)
        self.tf_params = cfg.transformer_params
        in_ch = 53 + cfg.opcode_embed_dim + self.tf_params.config_hidden_dim

        self.transformer = Transformer(
            31 - cfg.layout_override,
            self.tf_params.config_hidden_dim,
            self.tf_params.n_layers,
            self.tf_params.n_head,
        )

        if cfg.use_pre_linear:
            self.pre_linear = torch.nn.Sequential(torch.nn.Linear(in_ch, cfg.mid_ch), torch.nn.ReLU())
        self.convs = torch.nn.ModuleList()
        self.rev_convs = torch.nn.ModuleList()
        for i in range(cfg.n_layer):
            tmp_in = cfg.mid_ch if i > 0 or cfg.use_pre_linear else in_ch
            self.convs.append(get_graph_conv(cfg.conv_type, tmp_in, cfg.mid_ch // 2))
            self.rev_convs.append(get_graph_conv(cfg.conv_type, tmp_in, cfg.mid_ch // 2))
        self.head = torch.nn.Linear(cfg.mid_ch * cfg.concat_last_n, 1)

    def _forward_transformer(self, layout_feat: torch.Tensor) -> torch.Tensor:
        """(num_nodes * batch_size, L, in_dim) -> (num_nodes * batch_size, L, config_hidden_dim)"""
        pos_emb = torch.arange(6, device=layout_feat.device)[:, None]
        transformer_input = torch.cat([layout_feat, pos_emb[None].expand(layout_feat.shape[0], -1, -1)], dim=2)
        # (*, 4, 6 + 2)
        unique_input, rev_idx = transformer_input.unique(sorted=False, return_inverse=True, dim=0)
        transformer_output = self.transformer(unique_input).mean(1)[rev_idx]
        return transformer_output

    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """(num_nodes * batch_size, in_ch) -> (num_nodes * batch_size, 1)"""
        if self.cfg.use_pre_linear:
            x = self.pre_linear(x)
        drop_edge_ratio = random.uniform(0.0, self.cfg.drop_edge_ratio_max)
        edge_index = dropout_edge(edge_index, p=drop_edge_ratio, training=self.training)[0]
        rev_edge_index = torch.flip(edge_index, (0,))
        xs = []
        for i in range(len(self.convs)):
            x = torch.cat(
                [
                    self.convs[i](x, edge_index),
                    self.rev_convs[i](x, rev_edge_index),
                ],
                dim=-1,
            )
            x = F.relu(x)
            if i >= len(self.convs) - self.cfg.concat_last_n:
                xs.append(x)
        return torch.cat(xs, dim=-1)

    def forward(self, data: Batch) -> torch.Tensor:
        # (batch_size, num_nodes * batch_size, in_ch) -> (batch_size, sample_per_graph)
        num_nodes, sample_per_graph, _, _ = data.layout_feat.shape
        if "gat" in self.cfg.conv_type:
            raise ValueError("GAT is not supported")
        opcode_embed = self.opcode_embed(data.node_opcode)

        # Reverse layout feature
        mask1 = torch.where(data.layout_feat != -1)
        mask2 = (*mask1[:-1], data.layout_feat[mask1].to(torch.long))
        layout_rev_feat = torch.full_like(data.layout_feat, -1)
        layout_rev_feat[mask2] = mask1[-1].to(data.layout_feat.dtype)
        dim_feat = torch.cat(
            [
                data.dim_feat[None, :, :, :].expand(sample_per_graph, -1, -1, -1),
                layout_rev_feat.transpose(0, 1).transpose(-1, -2),
                data.tile_feat[data.batch].transpose(0, 1).transpose(-1, -2),
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (sample_per_graph * num_nodes * batch_size, 6, n_dim_feat)
        transformer_output = self._forward_transformer(dim_feat)
        config_embed = transformer_output.view(sample_per_graph, data.num_nodes, self.tf_params.config_hidden_dim)

        x = torch.cat(
            [
                data.x[None].expand(sample_per_graph, -1, -1),
                opcode_embed[None].expand(sample_per_graph, -1, -1),
                config_embed,
            ],
            dim=2,
        )  # (sample_per_graph, num_nodes * batch_size, 140 + opcode_embed_dim + 18)
        x = self._forward_gnn(x, data.edge_index, data.batch)
        return self.head(global_add_pool(x, data.batch, len(data.y))).transpose(0, 1)
