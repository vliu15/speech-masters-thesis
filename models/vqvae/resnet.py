"""Implements various ResNet layers"""

import math

import torch
import torch.nn as nn


def get_mod_cycle(depth, cycle):
    if cycle is None:
        return depth
    else:
        return depth % cycle


class ResLayer(nn.Module):

    def __init__(self, n_in, n_state, n_out=None, dilation=1, kernel_size=3, zero_out=True, res_scale=1.0):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        if n_out is None:
            n_out = n_in
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, kernel_size, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_out, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class ResNetBlock(nn.Module):
    """
    1D variation of the standard residual block

    Reference:
        - Deep Residual Learning for Image Recognition (2015)
    """

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
        **kwargs,
    ):
        super().__init__()
        blocks = [
            ResLayer(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate**get_mod_cycle(depth, dilation_cycle),
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


class HiFiBlock(nn.Module):
    """
    Multi-Resolution Fusion kernel from HiFi-GAN vocoder

    Reference:
        - HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (2020)
    """

    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        kernel_size_growth_rate=2,
        kernel_size_cycle=None,
        zero_out=True,
        res_scale=False,
        **kwargs,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResLayer(
                    n_in,
                    int(m_conv * n_in),
                    dilation=dilation_growth_rate**get_mod_cycle(depth, dilation_cycle),
                    kernel_size=3 + kernel_size_growth_rate * get_mod_cycle(depth, kernel_size_cycle),
                    zero_out=zero_out,
                    res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)
                ) for depth in range(n_depth)
            ]
        )

    def forward(self, x, mask=None):
        mask = 1. if mask is None else mask
        x = [block(x * mask) for block in self.blocks]
        return sum(x) / len(x), mask


class WaveNetBlock(nn.Module):
    """
    Stack of WaveNet residual layers

    Reference:
        - WaveNet: A Generative Model for Raw Audio (2016)
    """

    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=True,
        res_scale=False,
        **kwargs,
    ):
        super().__init__()
        n_hid = int(m_conv * n_in)
        self.res_scale = 1.0 if not res_scale else 1.0 / math.sqrt(n_depth)

        self.conv_in = nn.Conv1d(n_in, n_hid, 1)
        self.conv_out = nn.Conv1d(n_hid, n_in, 1)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    n_hid,
                    2 * n_hid,
                    3,
                    padding=dilation_growth_rate**get_mod_cycle(depth, dilation_cycle),
                    dilation=dilation_growth_rate**get_mod_cycle(depth, dilation_cycle),
                ) for depth in range(n_depth)
            ]
        )
        self.gates = nn.ModuleList([nn.Conv1d(
            n_hid,
            n_hid,
            1,
        ) for _ in range(n_depth)])
        if zero_out:
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    def forward(self, x, mask=None):
        mask = 1. if mask is None else mask

        x = self.conv_in(x * mask)
        for conv, gate in zip(self.convs, self.gates):
            z = conv(x * mask)
            t, s = z.chunk(2, dim=1)
            z = torch.tanh(t) * torch.sigmoid(s)
            z = gate(z * mask)
            x = x + self.res_scale * z
        x = self.conv_out(x * mask)
        return x, mask


class GatedHiFiBlock(nn.Module):
    """Fusion of HiFiBlock and WaveNetBlock"""

    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        kernel_size_growth_rate=2,
        kernel_size_cycle=None,
        zero_out=True,
        res_scale=False,
        **kwargs,
    ):
        super().__init__()

        n_hid = int(m_conv * n_in)
        self.res_scale = 1.0 if not res_scale else 1.0 / math.sqrt(n_depth)

        self.conv_in = nn.Conv1d(n_in, n_hid, 1)
        self.conv_out = nn.Conv1d(n_hid, n_in, 1)

        if zero_out:
            nn.init.zeros_(self.conv_out.weight)
            nn.init.zeros_(self.conv_out.bias)

        self.blocks = nn.ModuleList(
            [
                ResLayer(
                    n_hid,
                    n_hid,
                    n_out=2 * n_hid,
                    dilation=dilation_growth_rate**get_mod_cycle(depth, dilation_cycle),
                    kernel_size=3 + kernel_size_growth_rate * get_mod_cycle(depth, kernel_size_cycle),
                    zero_out=zero_out,
                    res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)
                ) for depth in range(n_depth)
            ]
        )

    def forward(self, x, mask=None):
        mask = 1. if mask is None else mask

        x = self.conv_in(x * mask)

        # Apply HiFiBlock layers and split into pre-gating activations
        t, s = [], []
        for block in self.blocks:
            z = block(x * mask)
            z_t, z_s = z.chunk(2, dim=1)
            t += [z_t]
            s += [z_s]

        # Unsqueeze on extra dimension to apply softmax/tanh gating, then aggregate over this dimension
        t = torch.stack(t, dim=1)
        s = torch.stack(s, dim=1)
        z = torch.tanh(t) * torch.softmax(s, dim=1)
        z = torch.sum(z, dim=1)

        z = self.conv_out(z * mask)
        x = x + self.res_scale(z)
        return x, mask
