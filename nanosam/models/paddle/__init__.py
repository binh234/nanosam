import math
from .backbone.pp_lcnet_v2 import (
    PPLCNetV2Backbone_small,
    PPLCNetV2Backbone_base,
    PPLCNetV2Backbone_large,
)
from .backbone.pp_hgnet import PPHGNetBackbone_tiny, PPHGNetBackbone_small
from .backbone.pp_hgnet_v2 import (
    PPHGNetV2Backbone_B0,
    PPHGNetV2Backbone_B1,
    PPHGNetV2Backbone_B2,
    PPHGNetV2Backbone_B3,
    PPHGNetV2Backbone_B4,
    PPHGNetV2Backbone_B5,
    PPHGNetV2Backbone_B6,
)
from .image_encoder import PaddleImageEncoder
from .neck import ConvNeck, SamNeck


def Sam_PPHGNetV2_B0(**kwargs):
    backbone = PPHGNetV2Backbone_B0(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B1(**kwargs):
    backbone = PPHGNetV2Backbone_B1(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B2(**kwargs):
    backbone = PPHGNetV2Backbone_B2(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1536, 768, 384],
        head_width=256,
        head_depth=6,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B3(**kwargs):
    backbone = PPHGNetV2Backbone_B3(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=6,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B4(**kwargs):
    backbone = PPHGNetV2Backbone_B4(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B5(**kwargs):
    backbone = PPHGNetV2Backbone_B5(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNetV2_B6(**kwargs):
    backbone = PPHGNetV2Backbone_B6(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[2048, 1024, 512],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNet_tiny(**kwargs):
    backbone = PPHGNetBackbone_tiny(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 512, 448],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPHGNet_small(**kwargs):
    backbone = PPHGNetBackbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 768, 512],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2_small(**kwargs):
    backbone = PPLCNetV2Backbone_small(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[768, 384, 192],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2_base(**kwargs):
    backbone = PPLCNetV2Backbone_base(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
    return model


def Sam_PPLCNetV2large(**kwargs):
    backbone = PPLCNetV2Backbone_large(**kwargs)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[1280, 640, 320],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmbconv",
    )

    model = PaddleImageEncoder(backbone, neck)
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
        neck_channels=256,
        pos_embedding=True,
    )

    model = PaddleImageEncoder(backbone, neck)
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
        neck_channels=256,
        pos_embedding=True,
    )

    model = PaddleImageEncoder(backbone, neck)
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
        neck_channels=256,
        pos_embedding=True,
    )

    model = PaddleImageEncoder(backbone, neck)
    return model
