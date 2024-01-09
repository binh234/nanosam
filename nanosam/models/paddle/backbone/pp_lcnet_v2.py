import paddle
from functools import partial
from ppcls.arch.backbone import PPLCNetV2_base, PPLCNetV2_large, PPLCNetV2_small
from typing import Dict


def PPLCNetV2Backbone__forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
    output_dict = {}
    x = self.stem(x)
    x = self.pool(x)

    for stage_id, stage in enumerate(self.stages, 1):
        output_dict["stage%d" % stage_id] = x = stage(x)
    return x


def PPLCNetV2Backbone_small(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNetV2Backbone_small
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPLCNetV2Backbone_small` model depends on args.
    """

    model = PPLCNetV2_small(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    model.forward = partial(PPLCNetV2Backbone__forward, model)
    return model


def PPLCNetV2Backbone_base(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNetV2Backbone_base
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPLCNetV2Backbone_base` model depends on args.
    """

    model = PPLCNetV2_base(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    model.forward = partial(PPLCNetV2Backbone__forward, model)
    return model


def PPLCNetV2Backbone_large(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNetV2Backbone_large
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPLCNetV2Backbone_large` model depends on args.
    """

    model = PPLCNetV2_large(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    model.forward = partial(PPLCNetV2Backbone__forward, model)
    return model
