import copy
import math
import paddle
import paddle.nn as nn
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.arch.slim import prune_model, quantize_model
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
from .nn.norm import LayerNorm2D


def build_model(config, mode="train"):
    arch_config = copy.deepcopy(config["Arch"])
    model_type = arch_config.pop("name")
    model = eval(model_type)(**arch_config)

    if isinstance(model, TheseusLayer):
        prune_model(config, model)
        quantize_model(config, model, mode)
    return model


class ImageEncoder(TheseusLayer):
    def __init__(
        self,
        backbone: Union[Dict, nn.Layer],
        neck: Union[Dict, nn.Layer],
        use_last_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone if isinstance(backbone, nn.Layer) else build_model(backbone)
        self.neck = neck if isinstance(neck, nn.Layer) else build_model(neck)
        self.use_last_norm = use_last_norm
        if self.use_last_norm:
            self.norm = LayerNorm2D(256, use_layernorm_op=kwargs.get("use_layernorm_op", False))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        feed_dict = self.backbone(x)
        out = self.neck(feed_dict)

        if isinstance(out, dict):
            out = out["features"]

        if self.use_last_norm:
            out = self.norm(out)

        return out


def Sam_PPHGNetV2_B0(
    middle_op="hgv2",
    head_depth=4,
    expand_ratio=1,
    num_block=2,
    use_last_norm=True,
    use_lab=True,
    **kwargs,
):
    backbone = PPHGNetV2Backbone_B0(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=use_lab,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B1(
    middle_op="hgv2",
    head_depth=4,
    expand_ratio=1,
    num_block=2,
    use_last_norm=True,
    use_lab=True,
    **kwargs,
):
    backbone = PPHGNetV2Backbone_B1(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=use_lab,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B2(
    middle_op="hgv2",
    head_depth=4,
    expand_ratio=1,
    num_block=2,
    use_last_norm=True,
    use_lab=True,
    **kwargs,
):
    backbone = PPHGNetV2Backbone_B2(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1536, 768, 384],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=use_lab,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B3(
    middle_op="hgv2",
    head_depth=5,
    expand_ratio=1,
    num_block=2,
    use_last_norm=True,
    use_lab=True,
    **kwargs,
):
    backbone = PPHGNetV2Backbone_B3(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=use_lab,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B4(
    middle_op="hgv2", head_depth=6, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPHGNetV2Backbone_B4(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B5(
    middle_op="hgv2", head_depth=6, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPHGNetV2Backbone_B5(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNetV2_B6(
    middle_op="hgv2", head_depth=6, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPHGNetV2Backbone_B6(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
        use_lab=False,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNet_tiny(
    middle_op="hgv2", head_depth=4, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPHGNetBackbone_tiny(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 512, 448],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPHGNet_small(
    middle_op="hgv2", head_depth=6, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPHGNetBackbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 768, 512],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPLCNetV2_small(
    middle_op="repdw", head_depth=4, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPLCNetV2Backbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 384, 192],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPLCNetV2_base(
    middle_op="repdw", head_depth=4, expand_ratio=1, num_block=1, use_last_norm=True, **kwargs
):
    backbone = PPLCNetV2Backbone_base(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Sam_PPLCNetV2_large(
    middle_op="repdw", head_depth=8, expand_ratio=1, num_block=2, use_last_norm=True, **kwargs
):
    backbone = PPLCNetV2Backbone_large(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1280, 640, 320],
        head_width=256,
        head_depth=head_depth,
        num_block=num_block,
        expand_ratio=expand_ratio,
        middle_op=middle_op,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Conv_PPHGNet_tiny(image_size=512, use_last_norm=False, head_depth=3, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPHGNetBackbone_tiny(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=768,
        num_upsample=num_upsample,
        head_depth=head_depth,
        mid_channels=256,
        out_channels=256,
        pos_embedding=True,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Conv_PPHGNetV2_B1(image_size=512, use_last_norm=False, head_depth=3, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPHGNetV2Backbone_B1(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=1024,
        num_upsample=num_upsample,
        head_depth=head_depth,
        mid_channels=256,
        out_channels=256,
        pos_embedding=True,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Conv_PPHGNetV2_B4(image_size=512, use_last_norm=False, head_depth=4, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPHGNetV2Backbone_B4(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=2048,
        num_upsample=num_upsample,
        head_depth=head_depth,
        mid_channels=256,
        out_channels=256,
        pos_embedding=True,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model


def Conv_PPLCNetV2_base(image_size=512, use_last_norm=False, head_depth=3, **kwargs):
    num_upsample = int(math.log2(2048 / image_size))
    backbone = PPLCNetV2Backbone_base(**kwargs)
    neck = ConvNeck(
        fid="stage_final",
        in_channels=1024,
        num_upsample=num_upsample,
        head_depth=head_depth,
        mid_channels=256,
        out_channels=256,
        pos_embedding=True,
    )

    model = ImageEncoder(backbone, neck, use_last_norm=use_last_norm, **kwargs)
    return model
