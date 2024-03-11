from ppcls.arch.backbone.legendary_models.pp_hgnet_v2 import HGV2_Block
from ppcls.arch.backbone.legendary_models.pp_lcnet_v2 import RepDepthwiseSeparable
from typing import List

from ..nn.ops import ConvLayer, DAGBlock, FusedMBConv, OpSequential, UpSampleLayer, HGV2_Act_Block, LCNetV3Block


class SamNeck(DAGBlock):
    def __init__(
        self,
        fid_list: List[str],
        in_channel_list: List[int],
        head_width: int,
        head_depth: int,
        num_block: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm="bn2d",
        act_func="gelu",
        use_lab=False,
        **kwargs,
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(
                        in_channel, head_width, 1, norm=None, act_func=None
                    ),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        if middle_op == "fmb":
            middle = []
            for _ in range(head_depth):
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                    identity=True,
                )
                middle.append(block)
            middle = OpSequential(middle)
        elif middle_op == "repdw":
            middle = []
            for _ in range(head_depth):
                block = RepDepthwiseSeparable(
                    in_channels=head_width,
                    out_channels=head_width,
                    stride=1,
                    dw_size=3,
                    split_pw=False,
                    use_rep=True,
                    use_se=False,
                    use_shortcut=True,
                )
                middle.append(block)
            middle = OpSequential(middle)
        elif middle_op == "lcv3":
            layer_num = head_depth // num_block
            middle = []
            for _ in range(num_block):
                block = LCNetV3Block(
                    in_channels=head_width,
                    out_channels=head_width,
                    conv_kxk_num=layer_num,
                    stride=1,
                    dw_size=5,
                    use_se=False,
                )
                middle.append(block)
            middle = OpSequential(middle)
        elif middle_op == "hgv2_act":
            layer_num = head_depth // num_block
            middle = []
            for _ in range(num_block):
                block = HGV2_Act_Block(
                    in_channels=head_width,
                    mid_channels=round(head_width * expand_ratio),
                    out_channels=head_width,
                    kernel_size=3,
                    layer_num=layer_num,
                    identity=True,
                    light_block=True,
                    norm=norm,
                    act_func=act_func,
                    use_lab=use_lab,
                )
                middle.append(block)
            middle = OpSequential(middle)
        elif middle_op == "hgv2":
            layer_num = head_depth // num_block
            middle = []
            for _ in range(num_block):
                block = HGV2_Block(
                    in_channels=head_width,
                    mid_channels=round(head_width * expand_ratio),
                    out_channels=head_width,
                    kernel_size=3,
                    layer_num=layer_num,
                    identity=True,
                    light_block=True,
                    use_lab=use_lab,
                )
                middle.append(block)
            middle = OpSequential(middle)
        else:
            raise NotImplementedError

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
