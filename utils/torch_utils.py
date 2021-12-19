import torch


def safe_log(x: torch.Tensor, eps=1e-8) -> None:
    return torch.log(torch.clamp(x, min=eps))
