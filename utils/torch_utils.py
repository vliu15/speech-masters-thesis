import torch


def safe_log(x: torch.Tensor, eps=1e-5) -> None:
    return torch.log(torch.clamp(x, min=eps))
