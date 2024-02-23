import torch.nn as nn

import timm
from typing import List

from .nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    HGV2Block,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
    build_norm,
)
from .registry import register_model


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
        norm="bn2d",
        act_func="gelu",
        **kwargs,
    ):
        inputs = {}
        for fid, in_channel in zip(fid_list, in_channel_list):
            inputs[fid] = OpSequential(
                [
                    ConvLayer(in_channel, head_width, 1, norm=None, act_func=None),
                    UpSampleLayer(size=(64, 64)),
                ]
            )

        if middle_op == "mb":
            middle = []
            for _ in range(head_depth):
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
                middle.append(ResidualBlock(block, IdentityLayer()))
            middle = OpSequential(middle)
        elif middle_op == "fmb":
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
                middle.append(ResidualBlock(block, IdentityLayer()))
            middle = OpSequential(middle)
        elif middle_op == "hgv2":
            middle = HGV2Block(
                in_channels=head_width,
                mid_channels=round(head_width * expand_ratio),
                out_channels=head_width,
                kernel_size=3,
                layer_num=head_depth,
                identity=True,
                light_block=True,
                norm=norm,
                act_func=act_func,
            )
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


class SamImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        middle_op: str = "hgv2",
        head_depth: int = 6,
        expand_ratio: int = 1,
        norm: str = "bn2d",
        act_func: str = "gelu",
        **kwargs,
    ):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        channels = self.backbone.feature_info.channels()
        self.fid_names = ["stage2", "stage3", "stage4"]

        print(channels)

        self.neck = SamNeck(
            fid_list=self.fid_names,
            in_channel_list=channels[-len(self.fid_names) :],
            head_width=256,
            head_depth=head_depth,
            expand_ratio=expand_ratio,
            middle_op=middle_op,
            norm=norm,
            act_func=act_func,
        )
        self.norm = build_norm("ln2d", 256)

    def forward(self, x):
        x = self.backbone(x)
        feat_dict = {}
        for name, feat in zip(self.fid_names, x[-len(self.fid_names) :]):
            feat_dict[name] = feat
        feat_dict = self.neck(feat_dict)

        output = feat_dict["features"]
        output = self.norm(output)
        return output


@register_model("efficientvit_b0_sam")
def efficientvit_b0(**kwargs):
    return SamImageEncoder("efficientvit_b0", pretrained=True, **kwargs)


@register_model("efficientvit_b1_sam")
def efficientvit_b1(**kwargs):
    return SamImageEncoder("efficientvit_b1", pretrained=True, **kwargs)


@register_model("efficientvit_b2_sam")
def efficientvit_b2(**kwargs):
    return SamImageEncoder("efficientvit_b2", pretrained=True, **kwargs)
