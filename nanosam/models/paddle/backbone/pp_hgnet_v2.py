import paddle
from ppcls.arch.backbone import (
    PPHGNetV2_B0,
    PPHGNetV2_B1,
    PPHGNetV2_B2,
    PPHGNetV2_B3,
    PPHGNetV2_B4,
    PPHGNetV2_B5,
    PPHGNetV2_B6,
)
from typing import Dict

from .wrapper import ModelWrapper


def PPHGNetV2Backbone__forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
    output_dict = {}
    x = self.stem(x)

    for stage_id, stage in enumerate(self.stages, 1):
        output_dict["stage%d" % stage_id] = x = stage(x)
    output_dict["stage_final"] = x
    return output_dict


def PPHGNetV2Backbone_B0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B0
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B0` model depends on args.
    """

    model = PPHGNetV2_B0(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B1(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B1
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B1` model depends on args.
    """

    model = PPHGNetV2_B1(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B2(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B2
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B2` model depends on args.
    """

    model = PPHGNetV2_B2(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B3(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B3
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B3` model depends on args.
    """

    model = PPHGNetV2_B3(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B4(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B4
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B4` model depends on args.
    """

    model = PPHGNetV2_B4(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B5
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B5` model depends on args.
    """

    model = PPHGNetV2_B5(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)


def PPHGNetV2Backbone_B6(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetV2Backbone_B6
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetV2Backbone_B6` model depends on args.
    """

    model = PPHGNetV2_B6(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetV2Backbone__forward)
