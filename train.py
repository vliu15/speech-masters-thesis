"""Script to run training in whole"""

import argparse
import logging
import logging.config
import os

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from utils.commons import get_dataloaders, get_model, get_optimizer, seed_all_rng
from utils.train_utils import train

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Name of model config class in configs/models")
    parser.add_argument(
        "--dataset", required=False, type=str, default="ljspeech", help="Name of dataset config class in configs/datasets"
    )
    parser.add_argument("--log_dir", required=False, type=str, default="./logs", help="Path to log directory")
    parser.add_argument("--seed", required=False, type=int, default=0, help="Seed for pseudo RNG")
    parser.add_argument("--batch_size", required=False, type=int, default=1, help="Batch size to use for training")

    parser.add_argument("--ema", required=False, default=False, action="store_true", help="Whether to track model EMA")
    parser.add_argument("--grad_clip_norm", required=False, type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--fp32", required=False, default=False, action="store_true", help="Run in FP32")

    parser.add_argument("--num_workers", required=False, type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--n_gpus", required=False, type=int, default=-1, help="Number of gpus to train on")
    parser.add_argument("--total_epochs", required=False, type=int, default=1000, help="Total epochs of training")
    parser.add_argument("--load_ckpt", required=False, type=str, default=None, help="Path to load checkpoint")

    parser.add_argument("--ckpt_every_n_steps", required=False, type=int, default=10000, help="Checkpointing step frequency")
    parser.add_argument("--log_every_n_steps", required=False, type=int, default=10, help="Logging step frequency")
    parser.add_argument("--eval_every_n_epochs", required=False, type=int, default=5, help="Validation epoch frequency")
    args = parser.parse_args()

    args.fp16 = not args.fp32
    return args


def train_multi(rank, world_size, config):
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
            from utils.train_hooks import run_ddi
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
        from utils.train_hooks import run_ddi
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

    if n_gpus <= 1:
        train_single(config)
    else:
        assert config.train.n_gpus <= max_gpus, f"Specified {config.train.n_gpus} gpus, but only {max_gpus} total"
        multiprocessing.spawn(train_multi, args=[config.train.n_gpus, config], nprocs=config.train.n_gpus, join=True)


if __name__ == "__main__":
    main()
