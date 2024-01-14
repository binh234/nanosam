import copy
import math
import paddle
import paddle.nn as nn
from typing import Dict, Union

from .backbone.pp_hgnet import PPHGNetBackbone_small, PPHGNetBackbone_tiny
from .backbone.pp_hgnet_v2 import (
    PPHGNetV2Backbone_B0,
    PPHGNetV2Backbone_B1,
    PPHGNetV2Backbone_B2,
    PPHGNetV2Backbone_B3,
    PPHGNetV2Backbone_B4,
    PPHGNetV2Backbone_B5,
    PPHGNetV2Backbone_B6,
)
from .backbone.pp_lcnet_v2 import (
    PPLCNetV2Backbone_base,
    PPLCNetV2Backbone_large,
    PPLCNetV2Backbone_small,
)
from .neck import ConvNeck, SamNeck


def build_model(config):
    arch_config = copy.deepcopy(config)
    model_type = arch_config.pop("name")
    model = eval(model_type)(**arch_config)
    return model


class ImageEncoder(nn.Layer):
    def __init__(
        self,
        backbone: Union[Dict, nn.Layer],
        neck: Union[Dict, nn.Layer],
        return_dict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone if isinstance(backbone, nn.Layer) else build_model(backbone)
        self.neck = neck if isinstance(neck, nn.Layer) else build_model(neck)
        self.return_dict = return_dict

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        feed_dict = self.backbone(x)
        out = self.neck(feed_dict)

        if not self.return_dict and isinstance(out, dict):
            out = out["features"]
        return out


def Sam_PPHGNetV2_B0(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPHGNetV2Backbone_B0(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=True,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B1(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPHGNetV2Backbone_B1(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=True,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B2(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPHGNetV2Backbone_B2(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1536, 768, 384],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=True,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B3(middle_op="fmbconv", head_depth=5, **kwargs):
    backbone = PPHGNetV2Backbone_B3(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=True,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B4(middle_op="fmbconv", head_depth=6, **kwargs):
    backbone = PPHGNetV2Backbone_B4(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B5(middle_op="fmbconv", head_depth=6, **kwargs):
    backbone = PPHGNetV2Backbone_B5(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B6(middle_op="fmbconv", head_depth=6, **kwargs):
    backbone = PPHGNetV2Backbone_B6(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNet_tiny(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPHGNetBackbone_tiny(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 512, 448],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPHGNet_small(middle_op="fmbconv", head_depth=6, **kwargs):
    backbone = PPHGNetBackbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 768, 512],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2_small(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPLCNetV2Backbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 384, 192],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2_base(middle_op="fmbconv", head_depth=4, **kwargs):
    backbone = PPLCNetV2Backbone_base(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2_large(middle_op="fmbconv", head_depth=6, **kwargs):
    backbone = PPLCNetV2Backbone_large(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1280, 640, 320],
        head_width=256,
        head_depth=head_depth,
        expand_ratio=1,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Conv_PPHGNet_tiny(image_size=512, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPHGNetBackbone_tiny(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=768,
        num_upsample=num_upsample,
        num_conv_layers=3,
        feature_dim=256,
        pos_embedding=True,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Conv_PPHGNetV2_B1(image_size=512, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPHGNetV2Backbone_B1(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=1024,
        num_upsample=num_upsample,
        num_conv_layers=3,
        feature_dim=256,
        pos_embedding=True,
        use_lab=True,
    )

    model = ImageEncoder(backbone, neck)
    return model


def Conv_PPLCNetV2_base(image_size=512, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPLCNetV2Backbone_base(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=1024,
        num_upsample=num_upsample,
        num_conv_layers=3,
        feature_dim=256,
        pos_embedding=True,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck)
    return model
