import torch
import torch.nn as nn


class MaskedConv1d(nn.Conv1d):

    def forward(self, x, mask):
        x = super().forward(x * mask)
        mask = mask[:, :, ::self.stride[0]]
        return x, mask


class MaskedConvTranspose1d(nn.ConvTranspose1d):

    def forward(self, x, mask):
        x = super().forward(x * mask)
        mask = mask.repeat_interleave(self.stride[0], dim=-1)
        return x, mask


def get_block(block_type):
    if block_type == "base":
        from models.vqvae.resnet import ResNetBlock
        return ResNetBlock
    elif block_type == "wavenet":
        from models.vqvae.resnet import WaveNetBlock
        return WaveNetBlock
    elif block_type == "hifi":
        from models.vqvae.resnet import HiFiBlock
        return HiFiBlock
    elif block_type == "gated_hifi":
        from models.vqvae.resnet import GatedHiFiBlock
        return GatedHiFiBlock
    else:
        raise ValueError(f"Didn't recognize block_type={block_type}")


class EncoderConvBlock(nn.Module):

    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        block_type,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        kernel_size_growth_rate=2,
        kernel_size_cycle=None,
        zero_out=True,
        res_scale=False,
    ):
        super().__init__()
        self.stride_t = stride_t
        Block = get_block(block_type)
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                blocks += [MaskedConv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t)]
                blocks += [
                    Block(
                        width,
                        depth,
                        m_conv=m_conv,
                        dilation_growth_rate=dilation_growth_rate,
                        dilation_cycle=dilation_cycle,
                        kernel_size_growth_rate=kernel_size_growth_rate,
                        kernel_size_cycle=kernel_size_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                    )
                ]
            blocks += [MaskedConv1d(width, output_emb_width, 3, 1, 1)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for block in self.blocks:
            x, mask = block(x, mask)
        return x, mask


class DecoderConvBlock(nn.Module):

    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        block_type,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        kernel_size_growth_rate=2,
        kernel_size_cycle=None,
        zero_out=True,
        res_scale=False,
        reverse_decoder_dilation=False,
    ):
        super().__init__()
        self.stride_t = stride_t
        Block = get_block(block_type)
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            blocks += [MaskedConv1d(output_emb_width, width, 3, 1, 1)]
            for i in range(down_t):
                blocks += [
                    Block(
                        width,
                        depth,
                        m_conv=m_conv,
                        dilation_growth_rate=dilation_growth_rate,
                        dilation_cycle=dilation_cycle,
                        kernel_size_growth_rate=kernel_size_growth_rate,
                        kernel_size_cycle=kernel_size_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation
                    )
                ]
                blocks += [
                    MaskedConvTranspose1d(
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
            x, mask = block(x, mask)
        return x, mask
