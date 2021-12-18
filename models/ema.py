"""Implements Exponential Moving Average modules"""

import logging

import torch.nn as nn

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class DummyEMA(nn.Module):
    """Dummy Exponential Moving Average module to simplify logic in training loop."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("EMA doesn't have a forward pass")

    def step(self):
        pass

    def swap(self):
        pass


class EMA(nn.Module):
    """Exponential Moving Average module for model weights."""

    def __init__(self, model, mu=0.99):
        super().__init__()
        self.mu = mu
        self.state = {n: [p, self.get_model_state(p)] for n, p in model.named_parameters() if p.requires_grad}

    def forward(self, *args, **kwargs):
        raise NotImplementedError("EMA doesn't have a forward pass")

    def state_dict(self):
        return {n: state[1] for n, state in self.state.items()}

    def load_state_dict(self, state_dict):
        # Check key coverage
        not_found_in_ckpt = sorted(set(self.state.keys()) - set(state_dict.keys()))
        if len(not_found_in_ckpt):
            not_found_in_ckpt = ["\n" + key for key in not_found_in_ckpt]
            logger.warning("Found %s parameters not in state_dict: %s", len(not_found_in_ckpt), ''.join(not_found_in_ckpt))

        # Load keys
        for n, state in state_dict.items():
            assert n in self.state, f"Parameter {n} found in state_dict but not in model EMA"
            assert self.state[n][0].shape == state.shape, \
                f"Parameter {n} expected to have shape {self.state[n][0].shape}, found shape {state.shape} in state_dict"
            self.state[n][1] = state

    def get_model_state(self, p):
        return p.data.float().detach().clone()

    def step(self):
        for n in self.state:
            p, state = self.state[n]
            state.mul_(self.mu).add_(1 - self.mu, p.data.float())

    def swap(self):
        # swap ema and model params
        for n in self.state:
            p, state = self.state[n]
            other_state = self.get_model_state(p)
            p.data.copy_(state.type_as(p.data))
            state.copy_(other_state)
