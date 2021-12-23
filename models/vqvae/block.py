import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqvae.resnet import Resnet1D

class EncoderConvBlock(nn.Module):

    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=True,
        res_scale=False,
    ):
        super().__init__()
        self.stride_t = stride_t
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                blocks += [nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t)]
                blocks += [Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale)]
            blocks += [nn.Conv1d(width, output_emb_width, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for block in self.blocks:
            if isinstance(block, Resnet1D):
                x = block(x, mask)
            else:
                x = block(x * mask)
            if mask.shape[-1] != x.shape[-1]:
                mask = mask[:, :, ::self.stride_t]
        return x * mask, mask


class DecoderConvBlock(nn.Module):

    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=True,
        res_scale=False,
        reverse_decoder_dilation=False
    ):
        super().__init__()
        self.stride_t = stride_t
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            blocks += [nn.Conv1d(output_emb_width, width, 3, 1, 1)]
            for i in range(down_t):
                blocks += [
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation
                    )
                ]
                blocks += [
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if i == (down_t - 1) else width,
                        filter_t,
                        stride_t,
                        pad_t,
                    )
                ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for block in self.blocks:
            if isinstance(block, Resnet1D):
                x = block(x, mask)
            else:
                x = block(x * mask)
            if mask.shape[-1] != x.shape[-1]:
                mask = mask.repeat_interleave(self.stride_t, dim=-1)
        return x * mask, mask
