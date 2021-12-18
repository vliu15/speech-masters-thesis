"""Implements all custom learning rate schedulers"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class DummyLR(_LRScheduler):
    """Dummy LR Scheduler to simplify logic in training loop"""

    def get_lr(self):
        return self.base_lrs


class LinearWarmupLR(_LRScheduler):
    """LR Scheduler: linear warmup"""

    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1  # avoid division by 0
        scale = min(step / self.warmup_steps, 1)
        return [base_lr * scale for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """Noam Scheduler: linear warmup, anneal with factor 1/sqrt(step)"""

    def __init__(self, optimizer: Optimizer, dim_model: int, warmup_steps: int):
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1  # avoid division by 0
        scale = self.dim_model**(-0.5) * min(step**(-0.5), step * self.warmup_steps**(-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
