import logging
import logging.config
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from models.base import (
    SpectrogramReconstructionModel,
    TokenToSpectrogramModel,
    TokenToWaveformModel,
    WaveformReconstructionModel,
)
from utils.commons import barrier, print_top_level_summary, to_device
from utils.train_hooks import (
    accumulate_stats,
    log_stats,
    save_audio_and_computed_spect,
    save_checkpoint,
    save_spect_and_inverted_audio,
)

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def train_step(
    *,
    global_step: int,
    batch: Iterable[torch.Tensor],
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler = None,
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
    scaler: torch.cuda.amp.GradScaler = None,
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
                    config=config,
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
            break

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
    with torch.cuda.amp.autocast(enabled=config.train.fp16):
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
    """Runs one epoch of validation (noop if rank!=0)"""
    if rank != 0:
        return

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
                config=config,
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
    scaler = torch.cuda.amp.GradScaler() if config.train.fp16 else None
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
            if epoch % config.train.eval_every_n_epochs == 0:
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

            pbar.set_postfix(postfix)
            pbar.update(1)

    save_checkpoint(config, global_step, -1, model, ema, optimizer, scheduler)
    if rank == 0:
        writer.close()
