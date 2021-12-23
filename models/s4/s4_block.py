from models.s4.s4 import S4
from models.vqvae.block import EncoderConvBlock, DecoderConvBlock


class S4EncoderConvBlock(EncoderConvBlock):

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
        super()._init__(self,
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            m_conv,
            dilation_growth_rate=dilation_growth_rate,
            dilation_cycle=dilation_cycle,
            zero_out=zero_out,
            res_scale=res_scale,
        )
        self.s4_layer = S4(
            H=output_emb_width,
            l_max=None,
            d_state=width,
            dropout=0.1,
            transposed=True,
        )

    def forward(self, x, mask):
        x, mask = super().forward(x, mask)
        x = self.s4_layer(x * mask)
        return x, mask


class S4DecoderConvBlock(DecoderConvBlock):

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
        reverse_decoder_dilation=False,
    ):
        super().__init__(
            input_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            m_conv,
            dilation_growth_rate=dilation_growth_rate,
            dilation_cycle=dilation_cycle,
            zero_out=zero_out,
            res_scale=res_scale,
            reverse_decoder_dilation=reverse_decoder_dilation,
        )
        self.s4_layer = S4(
            H=input_emb_width,
            l_max=None,
            d_state=width,
            dropout=0.1,
            transposed=True,
        )

    def forward(self, x, mask):
        x, mask = super().forward(x, mask)
        x = self.s4_layer(x * mask)
        return x, mask
