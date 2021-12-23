import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_utils import safe_log


class BottleneckBlock(nn.Module):

    def __init__(self, k_bins, emb_width, mu, threshold):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.threshold = threshold
        self.reset_k()

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        self.register_buffer("k", torch.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_k(self, x):
        emb_width, k_bins = self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][:k_bins]
        if distributed.is_initialized():
            distributed.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = torch.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        emb_width, k_bins = self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = torch.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with torch.no_grad():
            # Calculate new centres
            x_l_onehot = torch.zeros(k_bins, x.shape[0], device=x.device)  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

            _k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            y = self._tile(x)
            _k_rand = y[torch.randperm(y.shape[0])][:k_bins]

            if distributed.is_initialized():
                distributed.broadcast(_k_rand, 0)
                distributed.all_reduce(_k_sum, distributed.ReduceOp.SUM)
                distributed.all_reduce(_k_elem, distributed.ReduceOp.SUM)

            # Update centres
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) \
                     + (1 - usage) * _k_rand

            _k_prob = _k_elem / torch.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -torch.sum(_k_prob * safe_log(_k_prob))  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x, mask):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        mask = mask.permute(0, 2, 1).contiguous()
        mask = mask.reshape(-1, 1)

        if x.shape[-1] == self.emb_width:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., :self.emb_width], x[..., self.emb_width:]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        return x, prenorm, mask

    def postprocess(self, x_l, x_d, x_shape, mask):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        mask = mask.reshape(N, T, 1).permute(0, 2, 1).contiguous()
        return x_l, x_d, mask

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.k.t()
        distance = torch.sum(
            x ** 2, dim=-1, keepdim=True
        ) - 2 * torch.matmul(x, k_w) + torch.sum(
            k_w ** 2, dim=0, keepdim=True
        )  # (N * L, b)
        min_distance, x_l = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return x_l, fit

    def dequantize(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x, mask):
        N, _, T = x.shape

        # Preprocess.
        x, _, _ = self.preprocess(x, mask)

        # Quantise
        x_l, _ = self.quantize(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantize
        x_d = self.dequantize(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, mask, update_k=True):
        N, _, T = x.shape

        # Preprocess
        x, prenorm, mask = self.preprocess(x, mask)
        indices = mask.bool()[:, 0]

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x[indices])

        # Quantise and dequantize through bottleneck
        with torch.no_grad():
            x_l, fit = self.quantize(x)
            x_d = self.dequantize(x_l)

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x[indices], x_l[indices])
        else:
            update_metrics = {}

        # Loss
        commit_loss = torch.norm(x_d[indices].detach() - x[indices]) ** 2 / (mask.sum() * x.shape[1])

        # Passthrough
        x_d = x + (x_d - x).detach()
        x_d = x_d * mask

        # Postprocess
        x_l, x_d, mask = self.postprocess(x_l, x_d, (N, T), mask)
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):

    def __init__(self, l_bins, emb_width, mu, levels, threshold):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu, threshold)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantized = [level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantized

    def forward(self, xs, x_masks):
        zs, xs_quantized, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            x, x_mask = xs[level], x_masks[level]
            z, x_quantized, commit_loss, metric = self.level_blocks[level](x, x_mask, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantized = x_quantized.detach()
            xs_quantized.append(x_quantized)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantized, commit_losses, metrics


class NoBottleneckBlock(nn.Module):

    def forward(self, x, mask, update_k=True):
        return x, x, 0, {}

    def restore_k(self):
        pass


class NoBottleneck(nn.Module):

    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for _ in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        return zs

    def forward(self, xs, x_masks):
        zero = torch.zeros((), device=xs[0].device)
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics
