import math

import torch.nn as nn


class ResConv1DBlock(nn.Module):

    def __init__(self, n_in, n_state, dilation=1, zero_out=True, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):

    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=True,
        res_scale=False,
        reverse_dilation=False,
    ):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate**_get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)
            ) for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.ModuleList(blocks)

    def forward(self, x, mask=None):
        mask = 1. if mask is None else mask
        for block in self.model:
            x = block(x * mask)
        return x, mask
