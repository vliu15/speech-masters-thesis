import torch


def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))
