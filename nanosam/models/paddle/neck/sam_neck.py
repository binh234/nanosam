from ppcls.arch.backbone.legendary_models.pp_hgnet_v2 import HGV2_Block
from typing import List

from ..nn.ops import ConvLayer, DAGBlock, FusedMBConv, OpSequential, UpSampleLayer


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: List[str],
        in_channel_list: List[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2D",
        act_func="gelu",
        use_lab=False,
        **kwargs,
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 3, norm=norm, act_func=act_func, use_lab=use_lab),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        middle = []
        for i in range(head_depth):
            if middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                    identity=True,
                )
            elif middle_op == "hgv2":
                block = HGV2_Block(
                    in_channels=head_width,
                    mid_channels=round(head_width * expand_ratio),
                    out_channels=head_width,
                    kernel_size=3,
                    layer_num=3,
                    identity=False if i == 0 else True,
                    light_block=True,
                    use_lab=use_lab,
                )
            else:
                raise NotImplementedError
            middle.append(block)
        middle = OpSequential(middle)

        outputs = {
            "features": OpSequential(
                [
                    ConvLayer(
                        head_width,
                        out_dim,
                        1,
                        use_bias=True,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SamNeck, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)
