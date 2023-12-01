from __future__ import annotations

import argparse
import glob
import hashlib
import os
import warnings
from typing import Iterator, Optional

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
import wandb
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch_geometric.loader import DataLoader

from config.config import Config, load_config
from src.dataset import GraphData, KaggleDataset
from src.nn import GNN, DimensionAttentionGNN


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training for Kaggle TPU Graph")
    parser.add_argument("--out_base_dir", default="result")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--exp_name", default="tmp")
    parser.add_argument("--project_name", default="kaggle-tpu-graph")
    parser.add_argument("--load_snapshot", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb_logger", action="store_true")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()


class TileLayoutSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        rng: np.random.Generator,
        dataset1: KaggleDataset,
        batch_size1: int,
        dataset2: KaggleDataset,
        batch_size2: int,
        n_epoch_split: int,
        epoch_idx: int,
        num_replicas: int,
        rank: int,
    ) -> None:
        self.rng = rng
        self.batch_size1 = batch_size1
        self.n_batch1 = len(dataset1) // batch_size1

        self.batch_size2 = batch_size2
        self.n_batch2 = len(dataset2) // batch_size2

        self.perm1 = rng.permutation(len(dataset1))
        self.perm2 = rng.permutation(len(dataset2)) + len(dataset1)

        # Split one epoch into `n_epoch_split`
        batch_perm = rng.permutation(self.n_batch1 + self.n_batch2)
        n_split = n_epoch_split * num_replicas
        n_each = (self.n_batch1 + self.n_batch2) // n_split
        idx = n_epoch_split * rank + epoch_idx
        self.batch_order = batch_perm[n_each * idx : n_each * (idx + 1)]
        print("tile:", self.n_batch1, "layout:", self.n_batch2)

    def __len__(self) -> int:
        return len(self.batch_order)

    def __iter__(self) -> Iterator[list[int]]:
        for batch_idx in self.batch_order:
            if batch_idx < self.n_batch1:
                idx = self.perm1[batch_idx * self.batch_size1 : (batch_idx + 1) * self.batch_size1]
            else:
                rem_idx = batch_idx - self.n_batch1
                idx = self.perm2[rem_idx * self.batch_size2 : (rem_idx + 1) * self.batch_size2]
            yield idx.tolist()


class KaggleDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: Config,
        seed: int,
        data_dir: str,
        exp_name: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.hash_offset = seed + int(hashlib.sha256(exp_name.encode()).hexdigest(), 16)

    def train_dataloader(self):
        real_epoch = self.trainer.current_epoch // self.cfg.n_epoch_split
        seed = self.hash_offset + real_epoch
        tile_paths = sorted(sum([glob.glob(f"{self.data_dir}/{data}") for data in self.cfg.tile_train_data], []))
        tile_dataset = KaggleDataset(
            tile_paths,
            self.cfg.tile_sample_per_graph,
            self.cfg,
            np.random.default_rng(seed),
            config_from_npy=False,
            dataset_type="tile",
        )
        layout_paths = sorted(sum([glob.glob(f"{self.data_dir}/{data}") for data in self.cfg.layout_train_data], []))
        layout_dataset = KaggleDataset(
            layout_paths, self.cfg.layout_sample_per_graph, self.cfg, np.random.default_rng(seed)
        )
        sampler = TileLayoutSampler(
            np.random.default_rng(seed),
            tile_dataset,
            self.cfg.tile_batch_size,
            layout_dataset,
            self.cfg.layout_batch_size,
            self.cfg.n_epoch_split,
            self.trainer.current_epoch % self.cfg.n_epoch_split,
            self.trainer.num_devices,
            self.trainer.global_rank,
        )
        return DataLoader(
            torch.utils.data.ConcatDataset([tile_dataset, layout_dataset]),
            batch_sampler=sampler,
            num_workers=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.trainer.max_epochs is None or self.trainer.current_epoch < self.cfg.n_epoch_split - 1:
            n_config_lim = self.cfg.small_n_config
        else:
            n_config_lim = -1
        layout_paths_list = [
            sorted(glob.glob(f"{self.data_dir}/layout/xla/random/valid/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/xla/default/valid/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/nlp/random/valid/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/nlp/default/valid/*.npz")),
        ]
        dls = [
            DataLoader(
                KaggleDataset(
                    layout_paths, self.cfg.layout_sample_per_graph, self.cfg, None, n_config_lim=n_config_lim
                ),
                batch_size=self.cfg.layout_batch_size,
                num_workers=2,
                persistent_workers=True,
            )
            for layout_paths in layout_paths_list
        ]
        tile_paths = sorted(glob.glob(f"{self.data_dir}/tile/xla/valid/*.npz"))
        dls.append(
            DataLoader(
                KaggleDataset(
                    tile_paths,
                    self.cfg.tile_sample_per_graph,
                    self.cfg,
                    None,
                    config_from_npy=False,
                    n_config_lim=n_config_lim,
                    dataset_type="tile",
                ),
                batch_size=self.cfg.tile_batch_size,
                num_workers=2,
                persistent_workers=True,
            )
        )
        return dls

    def test_dataloader(self):
        layout_paths_list = [
            sorted(glob.glob(f"{self.data_dir}/layout/xla/random/test/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/xla/default/test/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/nlp/random/test/*.npz")),
            sorted(glob.glob(f"{self.data_dir}/layout/nlp/default/test/*.npz")),
        ]
        dls = [
            DataLoader(
                KaggleDataset(layout_paths, self.cfg.layout_sample_per_graph, self.cfg, None),
                batch_size=self.cfg.layout_batch_size,
                num_workers=2,
                persistent_workers=True,
            )
            for layout_paths in layout_paths_list
        ]
        tile_paths = sorted(glob.glob(f"{self.data_dir}/tile/xla/test/*.npz"))
        dls.append(
            DataLoader(
                KaggleDataset(
                    tile_paths,
                    self.cfg.tile_sample_per_graph,
                    self.cfg,
                    None,
                    config_from_npy=False,
                    dataset_type="tile",
                ),
                batch_size=self.cfg.tile_batch_size,
                num_workers=2,
                persistent_workers=True,
            )
        )
        return dls


def pairwise_hinge(pred: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """pred: (..., n_compare), y: (..., n_compare)"""
    return ((y.unsqueeze(-1) > y.unsqueeze(-2)) * F.relu(margin - (pred.unsqueeze(-1) - pred.unsqueeze(-2)))).mean()


class KaggleModel(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        if cfg.model_name == "gnn":
            self.gnn: torch.nn.Module = GNN(cfg)
        elif cfg.model_name == "dimension-attention-gnn":
            self.gnn = DimensionAttentionGNN(cfg)
        else:
            raise ValueError(f"Unknown model name: {cfg.model_name}")
        self.out_dir = ""

    def forward(self, data: GraphData) -> torch.Tensor:
        """GraphData, (batch_size) -> (batch_size, sample_per_graph)"""
        return self.gnn(data).squeeze(2)

    def training_step(self, batch, batch_idx):
        data: GraphData = batch["data"]
        pred = self(data)
        assert batch["is_tile"].all() or not batch["is_tile"].any()
        if batch["is_tile"].any():
            loss = pairwise_hinge(pred, data.y, 0.1) * 0.5 + pairwise_hinge(pred.flatten(), data.y.flatten(), 0.1) * 0.5
        else:
            loss = pairwise_hinge(pred, data.y) * 0.5 + pairwise_hinge(pred.flatten(), data.y.flatten()) * 0.5
        return {
            "loss": loss,
            "pred": pred.cpu().detach(),
            "y": data.y.cpu().detach(),
            "mae": F.l1_loss(pred, data.y).cpu().detach(),
            "graph_idx": batch["graph_idx"].cpu().detach(),
            "config_idx": batch["config_idx"].cpu().detach(),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data: GraphData = batch["data"]
        pred = self(data)
        return {
            "loss": pairwise_hinge(pred, data.y).cpu().detach(),
            "pred": pred.cpu().detach(),
            "y": data.y.cpu().detach(),
            "mae": F.l1_loss(pred, data.y).cpu().detach(),
            "graph_idx": batch["graph_idx"].cpu().detach(),
            "config_idx": batch["config_idx"].cpu().detach(),
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data: GraphData = batch["data"]
        pred = self(data)
        return {
            "pred": pred.cpu().detach(),
            "graph_idx": batch["graph_idx"].cpu().detach(),
            "config_idx": batch["config_idx"].cpu().detach(),
        }

    def _gather_devices_and_steps(self, outputs: list[dict[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
        outputs = self.all_gather(outputs)
        assert self.trainer is not None
        if self.trainer.global_rank != 0 or len(outputs) == 0:
            return None

        epoch_results: dict[str, torch.Tensor] = {}
        for key in outputs[0].keys():
            if self.trainer.num_devices > 1:
                result = torch.cat(
                    [(x[key].unsqueeze(1) if x[key].dim() == 1 else x[key]) for x in outputs], dim=1
                ).flatten(end_dim=1)
            else:
                result = torch.cat([(x[key].unsqueeze(0) if x[key].dim() == 0 else x[key]) for x in outputs], dim=0)
            epoch_results[key] = result.detach().cpu()
        return epoch_results

    def _epoch_end(self, step_outputs: list[dict[str, torch.Tensor]], phase: str) -> dict:
        epoch_results = self._gather_devices_and_steps(step_outputs)
        if epoch_results is None:
            return {}

        n_sample = epoch_results["pred"].shape[1]
        orig_graph_idx = epoch_results["graph_idx"][:, None].expand(-1, n_sample).flatten()
        orig_config_idx = epoch_results["config_idx"].flatten()
        (graph_idx, config_idx), inv_idx = torch.unique(
            torch.stack([orig_graph_idx, orig_config_idx]), return_inverse=True, dim=1
        )
        pred = torch.zeros_like(graph_idx, dtype=epoch_results["pred"].dtype)
        pred[inv_idx] = epoch_results["pred"].flatten()
        if (
            self.trainer.max_epochs is None
            or ("val" in phase and self.trainer.current_epoch == self.trainer.max_epochs - 1)
            or "test" in phase
        ):
            np.savez(
                f"{self.out_dir}/{phase}.npz",
                pred=pred.numpy(),
                graph_idx=graph_idx.numpy(),
                config_idx=config_idx.numpy(),
            )

        if "test" in phase:
            return {}
        d = {
            f"{phase}/loss": epoch_results["loss"].mean().cpu(),
            f"{phase}/mae": epoch_results["mae"].mean().cpu(),
        }
        if "val" in phase:
            y = torch.zeros_like(graph_idx, dtype=epoch_results["y"].dtype)
            y[inv_idx] = epoch_results["y"].flatten()
            graph_idx = graph_idx.contiguous()
            boundaries = (graph_idx[1:] != graph_idx[:-1]).nonzero(as_tuple=False).flatten().tolist()
            assert len(boundaries) == graph_idx.max()
            boundaries = [0] + boundaries + [len(graph_idx)]
            taus = []
            for i in range(len(boundaries) - 1):
                left, right = boundaries[i], boundaries[i + 1]
                taus.append(scipy.stats.kendalltau(pred[left:right], y[left:right])[0])
            if np.nan in taus:
                print(f"warning: nan {taus.count(np.nan)} / {len(taus)}")
                taus = np.nan_to_num(taus).tolist()
            d[f"{phase}/kendalltau"] = np.mean(taus)
        print(d)
        self.log_dict(d, on_epoch=True)
        return d

    def training_epoch_end(self, training_step_outputs) -> None:
        pass

    def validation_epoch_end(self, validation_step_outputs):
        names = ["xla-random-val", "xla-default-val", "nlp-random-val", "nlp-default-val"]
        results = [self._epoch_end(validation_step_outputs[i], name) for i, name in enumerate(names)]
        self._epoch_end(validation_step_outputs[4], "tile-val")
        if len(results) == 0 or len(results[0]) == 0:
            return
        kendall_mean = np.mean([res[f"{name}/kendalltau"] for res, name in zip(results, names)])
        print(f"{kendall_mean=}")
        self.log("val/kendalltau-mean", kendall_mean, on_epoch=True)

    def test_epoch_end(self, test_step_outputs) -> None:
        names = ["xla-random-test", "xla-default-test", "nlp-random-test", "nlp-default-test", "tile-test"]
        for i, name in enumerate(names):
            self._epoch_end(test_step_outputs[i], name)

    def _get_total_steps(self) -> int:
        if not hasattr(self, "_total_steps"):
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            # Already divided by num_devices in TileLayoutSampler
            # accum = max(1, self.trainer.num_devices) * self.trainer.accumulate_grad_batches
            accum = self.trainer.accumulate_grad_batches
            self._total_steps = len(train_loader) // accum * self.trainer.max_epochs
        return self._total_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.lr, weight_decay=self.cfg.weight_decay)
        total_steps = self._get_total_steps()
        warmup_steps = round(total_steps * self.hparams.warmup_steps_ratio)
        print(f"lr warmup step: {warmup_steps} / {total_steps}")
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(args: argparse.Namespace, cfg: Config, seed: int) -> None:
    out_dir = f"{args.out_base_dir}/{args.exp_name}/{seed}"
    model = KaggleModel(cfg)
    model.out_dir = out_dir

    data_dir = f"{args.in_base_dir}/npz_all/npz"
    data_module = KaggleDataModule(cfg, seed, data_dir, args.exp_name)
    loggers: list[pl_loggers.Logger] = [pl_loggers.CSVLogger(out_dir)]
    if args.wandb_logger:
        loggers.append(
            pl_loggers.WandbLogger(
                project=args.project_name,
                group=args.exp_name,
                name=f"{args.exp_name}/{seed}",
                save_dir=out_dir,
            )
        )
    callbacks: list[Callback] = [LearningRateMonitor("epoch")]
    if args.save_checkpoint:
        callbacks.append(ModelCheckpoint(out_dir, save_last=True, save_top_k=0, every_n_epochs=1))
    n_gpus = torch.cuda.device_count()
    trainer = Trainer(
        gpus=n_gpus,
        max_epochs=cfg.max_epochs * cfg.n_epoch_split,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=args.save_checkpoint,
        precision=cfg.precision,
        gradient_clip_val=cfg.gradient_clip_val,
        strategy="ddp_find_unused_parameters_false" if n_gpus > 1 else None,
        reload_dataloaders_every_n_epochs=1,
        replace_sampler_ddp=False,
    )
    ckpt_path: Optional[str] = f"{out_dir}/last.ckpt"
    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=data_module)
    if args.save_model:
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
    with open(f"{out_dir}/config.yaml", "w") as f:
        yaml.dump(dict(cfg), f)
    trainer.test(model, datamodule=data_module)

    if args.wandb_logger:
        wandb.finish()


def main():
    args = parse()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    cfg = load_config(args.config_path, "config/default.yaml")
    print(cfg)
    train(args, cfg, 0)


if __name__ == "__main__":
    main()
