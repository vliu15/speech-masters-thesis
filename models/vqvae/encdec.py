import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, block_type, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        if block_type == "base":
            from models.vqvae.block import EncoderConvBlock as ConvBlock
        elif block_type == "s4":
            from models.s4.s4_block import S4EncoderConvBlock as ConvBlock
        else:
            raise ValueError(f"block_type={block_type} not recognized")

        block_kwargs_copy = dict(**block_kwargs)
        if "reverse_decoder_dilation" in block_kwargs_copy:
            del block_kwargs_copy["reverse_decoder_dilation"]
        level_block = lambda level, down_t, stride_t: ConvBlock(
            input_emb_width if level == 0 else output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs_copy
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x, x_mask):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert x.shape == (N, emb, T), f"Expected shape {(N, emb, T)}, got {x.shape}."

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            x, x_mask = self.level_blocks[level](x, x_mask)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert x.shape == (N, emb, T), f"Expected shape {(N, emb, T)}, got {x.shape}."

        return x, x_mask


class Decoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, block_type="base", **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        if block_type == "base":
            from models.vqvae.block import DecoderConvBlock as ConvBlock
        elif block_type == "s4":
            from models.s4.s4_block import S4DecoderConvBlock as ConvBlock
        else:
            raise ValueError(f"block_type={block_type} not recognized")

        level_block = lambda level, down_t, stride_t: ConvBlock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, x_masks, all_levels=True):
        if all_levels:
            assert len(xs) == len(x_masks) == self.levels
        else:
            assert len(xs) == len(x_masks) == 1
        x, x_mask = xs[-1], x_masks[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert x.shape == (N, emb, T), f"Expected shape {(N, emb, T)}, got {x.shape}."

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            x, x_mask = self.level_blocks[level](x, x_mask)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert x.shape == (N, emb, T), f"Expected shape {(N, emb, T)}, got {x.shape}."
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x * x_mask)
        return x * x_mask, x_mask
