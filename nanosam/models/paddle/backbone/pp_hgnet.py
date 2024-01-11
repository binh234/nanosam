import paddle
from ppcls.arch.backbone import PPHGNet_small, PPHGNet_tiny
from typing import Dict

from .wrapper import ModelWrapper


def PPHGNetBackbone__forward(self, x: paddle.Tensor) -> Dict[str, paddle.Tensor]:
    output_dict = {}
    x = self.stem(x)
    x = self.pool(x)

    for stage_id, stage in enumerate(self.stages, 1):
        output_dict["stage%d" % stage_id] = x = stage(x)
    output_dict["stage_final"] = x
    return output_dict


def PPHGNetBackbone_tiny(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetBackbone_tiny
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetBackbone_tiny` model depends on args.
    """

    model = PPHGNet_tiny(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetBackbone__forward)


def PPHGNetBackbone_small(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNetBackbone_small
    Args:
        pretrained (bool/str): If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld (bool) Whether using ssld pretrained model when pretrained is True.
    Returns:
        model: nn.Layer. Specific `PPHGNetBackbone_small` model depends on args.
    """

    model = PPHGNet_small(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    return ModelWrapper(model, PPHGNetBackbone__forward)
