"""Contains helper functions for Trainer"""

import logging
import logging.config
from typing import Iterable, Tuple

import torch
import torch.distributed as distributed
import torch.nn as nn
from omegaconf import DictConfig

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def to_device(batch: Iterable[torch.Tensor], device: str) -> None:
    """Puts a batch onto the specified device"""
    return [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]


def get_model(config: DictConfig, device: str = "cuda:0", rank: int = 0) -> Tuple[nn.Module, nn.Module]:
    """Returns initialized model and EMA"""
    import importlib
    if rank != 0:
        device = f"cuda:{rank}"

    # Init model
    model_file, model_class = config.model["_import_"].rsplit(".", 1)
    model = getattr(importlib.import_module(model_file), model_class)(config).to(device)

    # Surgery in preprocessing options for faster dataloading
    from models.base import (
        SpectrogramReconstructionModel,
        TokenToSpectrogramModel,
        TokenToWaveformModel,
        WaveformReconstructionModel,
    )
    if isinstance(model, (TokenToWaveformModel, WaveformReconstructionModel)):
        config.dataset.use_spect = False
    if isinstance(model, (TokenToSpectrogramModel, SpectrogramReconstructionModel)):
        config.dataset.use_audio = False
    if isinstance(model, (WaveformReconstructionModel, SpectrogramReconstructionModel)):
        config.dataset.use_token = False

    # Wrap model in DDP if initialized
    if distributed.is_initialized():
        # Convert BN -> SyncBN if applicable
        from torch.nn.modules.batchnorm import _BatchNorm
        if any(isinstance(module, _BatchNorm) for module in model.modules()):
            from torch.nn import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model)

        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=True)

    # Init EMA
    if not config.train.get("ema", False):
        from models.ema import DummyEMA
        ema = DummyEMA().to(device)
    else:
        from models.ema import EMA
        ema = EMA(model, mu=1 - (config.train.batch_size * config.train.n_gpus / 1000.)).to(device)

    return model, ema


def get_dataloaders(
    config: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns train and val dataloaders (val only for rank==0)"""
    import importlib

    # Init datasets
    dataset_file, dataset_class = config.dataset["_import_"].rsplit(".", 1)
    dataset = getattr(importlib.import_module(dataset_file), dataset_class)

    train_dataset = dataset(config, split="train")

    # Init (DDP) train dataloader
    from torch.utils.data import DataLoader
    if distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False),
            pin_memory=True,
            drop_last=False,
            collate_fn=dataset.collate,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=dataset.collate,
        )

    # Init (Rank 0) val dataloader
    if rank == 0:
        val_dataloader = DataLoader(
            dataset(config, split="val"),
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dataset.collate,
        )
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader


def get_optimizer(
    config: DictConfig,
    model: nn.Module,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Returns model optimizer and LR scheduler"""
    # Init optimizer
    if config.optimizer.name == "adam":
        from torch.optim import AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )
    elif config.optimizer.name == "sgd":
        from torch.optim import SGD
        optimizer = SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Didn't recognize optimizer name {config.optimizer.name}")

    # Init scheduler
    if not config.get("scheduler", None):
        from utils.lr_scheduler import DummyLR
        scheduler = DummyLR(optimizer)
    elif config.scheduler.name == "noam":
        from utils.lr_scheduler import NoamLR
        scheduler = NoamLR(
            optimizer,
            dim_model=config.model.encoder.hidden_channels,
            warmup_steps=config.scheduler.warmup_steps,
        )
    elif config.scheduler.name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=config.train.total_steps)
    elif config.scheduler.name == "linear":
        from utils.lr_scheduler import LinearWarmupLR
        scheduler = LinearWarmupLR(optimizer, warmup_steps=config.scheduler.warmup_steps)
    else:
        raise ValueError(f"Didn't recognize scheduler name {config.scheduler.name}")

    return optimizer, scheduler


def setup_logdir(config: DictConfig) -> None:
    """Initializes logs folder"""
    import os

    from omegaconf import OmegaConf

    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.train.log_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(config.train.log_dir, "spect"), exist_ok=True)
    os.makedirs(os.path.join(config.train.log_dir, "audio"), exist_ok=True)
    with open(os.path.join(config.train.log_dir, "config.yaml"), "w", encoding="utf-8") as f:
        OmegaConf.save(config=config, f=f.name)

    logger.info("Set up logdir at %s", config.train.log_dir)
