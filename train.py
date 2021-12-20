"""Script to run training in whole"""

import argparse
import logging
import logging.config
import os
from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.base import (
    SpectrogramReconstructionModel,
    TokenToSpectrogramModel,
    TokenToWaveformModel,
    WaveformReconstructionModel,
)
from utils.commons import get_dataloaders, get_model, get_optimizer, setup_logdir, to_device
from utils.train_utils import (
    accumulate_stats,
    barrier,
    log_stats,
    print_top_level_summary,
    save_audio_and_computed_spect,
    save_checkpoint,
    save_spect_and_inverted_audio,
    seed_all_rng,
)

# On rice.stanford.edu, only older versions of pytorch are supported
try:
    from torch.cuda.amp.autocast import GradScaler
except ModuleNotFoundError:
    GradScaler = None

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=False, type=str, default="vqvae", help="Name of model config class in configs/models"
    )
    parser.add_argument(
        "--dataset", required=False, type=str, default="ljspeech", help="Name of dataset config class in configs/datasets"
    )
    parser.add_argument("--log_dir", required=False, type=str, default="./logs/vqvae", help="Path to log directory")
    parser.add_argument("--seed", required=False, type=int, default=0, help="Seed for pseudo RNG")
    parser.add_argument("--batch_size", required=False, type=int, default=8, help="Batch size to use for training")

    parser.add_argument("--ema", required=False, default=False, action="store_true", help="Whether to track model EMA")
    parser.add_argument("--grad_clip_norm", required=False, type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--fp16", required=False, default=False, action="store_true", help="Run in FP16")

    parser.add_argument("--num_workers", required=False, type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--n_gpus", required=False, type=int, default=-1, help="Number of gpus to train on")
    parser.add_argument("--total_epochs", required=False, type=int, default=1000, help="Total epochs of training")
    parser.add_argument("--load_ckpt", required=False, type=str, default=None, help="Path to load checkpoint")

    parser.add_argument("--ckpt_every_n_steps", required=False, type=int, default=5000, help="Checkpointing step frequency")
    parser.add_argument("--log_every_n_steps", required=False, type=int, default=10, help="Logging step frequency")
    parser.add_argument("--eval_every_n_epochs", required=False, type=int, default=5, help="Validation epoch frequency")
    args = parser.parse_args()
    return args


def train_step(
    *,
    global_step: int,
    batch: Iterable[torch.Tensor],
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler = None,
    device: str,
    rank: int = 0,
):
    """Performs one step of forward pass, backpropagation, optimizer and EMA step"""
    batch = to_device(batch, device)
    optimizer.zero_grad()

    # Mixed precision: O1 forward pass, scale/clip gradients, schedule LR when no gradient overflow
    if config.train.fp16:
        with torch.cuda.amp.autocast(enabled=config.train.fp16):
            scaling_factor = scaler.get_scale()

            loss_dict, metrics_dict = model.supervised_step(batch)
            loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            # Optionally apply gradient clipping
            if config.train.grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            # Check for gradient overflow, if none then schedule LR
            if scaling_factor == scaler.get_scale():
                scheduler.step()
            else:
                logger.debug("[Rank %s] Gradient overflow detected. Loss scale lowered to %s", rank, scaler.get_scale())
                scaling_factor = scaler.get_scale()
        pass

    # Full precision: O0 forward pass with optional gradient clipping, schedule LR
    else:
        loss_dict, metrics_dict = model.supervised_step(batch)
        loss = loss_dict["loss"]
        if torch.isnan(loss):
            logger.info(
                dict(
                    **{k: loss_dict[k] for k in loss_dict.keys() if k.startswith("loss")},
                    **metrics_dict,
                    STEP=global_step,
                    RANK=rank,
                )
            )
            raise RuntimeError(f"Nan detected in loss at step {global_step}")
        loss.backward()
        # Optionally apply gradient clipping
        if config.train.grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
        optimizer.step()
        # Schedule LR
        scheduler.step()

    ema.step()
    return loss_dict, metrics_dict


def train_epoch(
    *,
    global_step: int,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    scaler: GradScaler = None,
    device: str,
    rank: int = 0,
):
    """Runs one epoch of standard neural network training"""
    postfix = {}
    losses, metrics = defaultdict(float), defaultdict(float)

    # Train epoch
    with tqdm(
            total=len(train_dataloader),
            leave=False,
            desc=f"Epoch {epoch} [train]",
            disable=(rank != 0),
    ) as pbar:

        model.train()
        for batch in train_dataloader:
            batch = to_device(batch, device)

            loss_dict, metrics_dict = train_step(
                global_step=global_step,
                batch=batch,
                config=config,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                rank=rank,
            )

            # Update per-rank stepwise averages
            global_step += 1
            pbar.update(1)

            # [Rank 0] Update loss averages, progress bars
            if rank == 0:
                # Update averages
                accumulate_stats(
                    over_n_steps=config.train.log_every_n_steps,
                    loss_dict=loss_dict,
                    metrics_dict=metrics_dict,
                    accumulated_loss=losses,
                    accumulated_metrics=metrics,
                )

                # Log and update progress bars
                if global_step % config.train.log_every_n_steps == 0:
                    log_stats(
                        step_or_epoch=global_step,
                        writer=writer,
                        losses=losses,
                        metrics=metrics,
                    )
                    postfix = dict(**losses, lr=optimizer.param_groups[0]["lr"])
                    pbar.set_postfix(postfix)
                    losses, metrics = defaultdict(float), defaultdict(float)

                # Save checkpoint
                if global_step % config.train.ckpt_every_n_steps == 0:
                    save_checkpoint(config, global_step, epoch, model, ema, optimizer, scheduler)

    return global_step, epoch + 1


def val_step(
    *,
    batch: Iterable[torch.Tensor],
    config: DictConfig,
    model: nn.Module,
    device: str,
):
    """Performs one validation step"""
    batch = to_device(batch, device)
    loss_dict, metrics_dict = model.supervised_step(batch)
    return loss_dict, metrics_dict


def val_epoch(
    *,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: str,
    rank: int = 0,
):
    """Runs one epoch of validation"""
    losses, metrics = defaultdict(float), defaultdict(float)

    y, yh = [], []
    with torch.no_grad():
        model.eval()
        ema.swap()

        for batch in tqdm(
                val_dataloader,
                total=len(val_dataloader),
                leave=False,
                desc=f"Epoch {epoch} [val]",
        ):
            loss_dict, metrics_dict = val_step(
                batch=batch,
                config=config,
                model=model,
                device=device,
            )

            # Accumulate losses, ground truths, and predictions
            accumulate_stats(
                over_n_steps=len(val_dataloader),
                loss_dict=loss_dict,
                metrics_dict=metrics_dict,
                accumulated_loss=losses,
                accumulated_metrics=metrics,
            )
            y += [loss_dict["y"].cpu()]
            yh += [loss_dict["yh"].cpu()]
        ema.swap()

    log_stats(
        step_or_epoch=epoch,
        writer=writer,
        losses=losses,
        metrics=metrics,
    )
    return torch.cat(y, dim=0).float().numpy(), torch.cat(yh, dim=0).float().numpy(), {**losses, **metrics}


def train(
    *,
    global_step: int,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: str,
    rank: int = 0,
):
    # Check that model has proper handle
    assert hasattr(model, "supervised_step") and callable(model.supervised_step), \
        f"Model type {model.__class__.__name__} doesn't have forward handle `supervised_step`"

    barrier()
    if rank == 0:
        print_top_level_summary(model)

    # Train
    postfix = {}
    scaler = GradScaler() if config.train.fp16 else None
    with tqdm(initial=epoch, total=config.train.total_epochs, desc="Global epoch", postfix=postfix,
              disable=(rank != 0)) as pbar:

        # Loop through epochs
        while True:
            if epoch >= config.train.total_epochs:
                break

            global_step, epoch = train_epoch(
                global_step=global_step,
                epoch=epoch,
                config=config,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=train_dataloader,
                writer=writer,
                scaler=scaler,
                device=device,
                rank=rank,
            )
            if epoch % config.train.eval_every_n_epochs == 0 and rank == 0:
                y, yh, postfix = val_epoch(
                    epoch=epoch,
                    config=config,
                    model=model,
                    ema=ema,
                    val_dataloader=val_dataloader,
                    writer=writer,
                    device=device,
                    rank=rank,
                )

                if isinstance(model, (TokenToWaveformModel, WaveformReconstructionModel)):
                    save_audio_and_computed_spect(config, epoch, writer, y, yh, n=4)
                elif isinstance(model, (TokenToSpectrogramModel, SpectrogramReconstructionModel)):
                    save_spect_and_inverted_audio(config, epoch, writer, y, yh, n=4)

            barrier()
            pbar.set_postfix(postfix)
            pbar.update(1)

    if rank == 0:
        save_checkpoint(config, global_step, -1, model, ema, optimizer, scheduler)
        writer.close()


## End of training helpers, train_multi for ddp and train_single for single xpu


def train_multi(rank, world_size, config):
    """Entry point into ddp training"""
    # RNG
    seed_all_rng(config.train.seed, cuda=True)

    # Initialize rank process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    distributed.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    device = torch.device(rank)
    torch.cuda.set_device(rank)

    # Init modules
    writer = SummaryWriter(config.train.log_dir) if rank == 0 else None
    model, ema = get_model(config, device, rank)
    optimizer, scheduler = get_optimizer(config, model)
    train_dataloader, val_dataloader = get_dataloaders(config, rank, world_size)

    # Run DDI if applicable to the model, but only 1 copy needs to do this
    if rank == 0:
        if config.model.get("ddi", False) and not config.train.load_ckpt:
            from utils.train_utils import run_ddi
            run_ddi(config, model, ema, optimizer, scheduler, train_dataloader, device)

            # Save the new config with checkpoint path so other ranks can load it
            with open(os.path.join(config.train.log_dir, "config.yaml"), "w", encoding="utf-8") as f:
                OmegaConf.save(config=config, f=f.name)

    # Wait here in case rank=0 is running DDI, then load config
    distributed.barrier()
    config = OmegaConf.load(os.path.join(config.train.log_dir, "config.yaml"))

    # Load checkpoint
    if config.train.load_ckpt:
        ckpt = torch.load(config.train.load_ckpt, map_location=torch.device(rank))
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        ema.load_state_dict(ckpt["ema"])
        global_step = ckpt["step"]
        epoch = ckpt["epoch"]
    else:
        global_step = 0
        epoch = 0

    logger.info(" [Rank %s / %s] Initialized all modules", rank, world_size)

    # Train
    try:
        distributed.barrier()
        train(
            global_step=global_step,
            epoch=epoch,
            config=config,
            model=model.module,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
            rank=rank,
        )
    except KeyboardInterrupt:
        pass

    # Destroy rank process
    distributed.destroy_process_group()


def train_single(config):
    """Entry point into single xpu training"""
    # RNG
    cuda = torch.cuda.is_available()
    seed_all_rng(config.train.seed, cuda=cuda)

    device = torch.device("cuda") if cuda else torch.device("cpu")

    # Init modules
    writer = SummaryWriter(config.train.log_dir)
    model, ema = get_model(config, device, 0)
    optimizer, scheduler = get_optimizer(config, model)
    train_dataloader, val_dataloader = get_dataloaders(config, 0, 1)

    # Run DDI if applicable to the model
    if config.model.get("ddi", False) and not config.train.load_ckpt:
        from utils.train_utils import run_ddi
        run_ddi(config, model, ema, optimizer, scheduler, train_dataloader, device)

    # Load checkpoint
    if config.train.load_ckpt:
        ckpt = torch.load(config.train.load_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        ema.load_state_dict(ckpt["ema"])
        global_step = ckpt["step"]
        epoch = ckpt["epoch"]
    else:
        global_step = 0
        epoch = 0

    logger.info("[%s] Initialized all modules", device)

    train(
        global_step=global_step,
        epoch=epoch,
        config=config,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        writer=writer,
        device=device,
        rank=0,
    )


def main():
    """Entry point into training"""
    args = parse_args()

    # Initialize model config
    model_config = OmegaConf.load(f"configs/models/{args.model}.yaml")

    # Initialize dataset config
    dataset_config = OmegaConf.load(f"configs/datasets/{args.dataset}.yaml")

    # Initialize train config
    train_config = OmegaConf.create(
        {
            "train":
                {
                    "log_dir": args.log_dir,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "ema": args.ema,
                    "grad_clip_norm": args.grad_clip_norm,
                    "fp16": args.fp16,
                    "num_workers": args.num_workers,
                    "n_gpus": args.n_gpus,
                    "total_epochs": args.total_epochs,
                    "load_ckpt": args.load_ckpt,
                    "ckpt_every_n_steps": args.ckpt_every_n_steps,
                    "log_every_n_steps": args.log_every_n_steps,
                    "eval_every_n_epochs": args.eval_every_n_epochs,
                }
        }
    )

    # Merge all configs
    config = OmegaConf.merge(model_config, dataset_config, train_config)

    max_gpus = torch.cuda.device_count()
    if config.train.n_gpus == -1:
        config.train.n_gpus = max_gpus

    # Determine whether to launch single or multi-gpu training
    n_gpus = min(config.train.n_gpus, max_gpus)
    if n_gpus == 0:
        logger.info("0 GPUs found. Using CPU.")
        config.train.n_gpus = 0
        config.train.fp16 = False

    # Set up log directory
    setup_logdir(config)

    # Train
    if n_gpus <= 1:
        logger.info("Training with 1 GPU.")
        train_single(config)
    else:
        assert config.train.n_gpus <= max_gpus, f"Specified {config.train.n_gpus} gpus, but only {max_gpus} total"
        logger.info("Training with %s GPUs.", config.train.n_gpus)
        multiprocessing.spawn(train_multi, args=[config.train.n_gpus, config], nprocs=config.train.n_gpus, join=True)


if __name__ == "__main__":
    main()
