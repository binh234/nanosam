import paddle
import paddle.nn as nn

from .utils import build_kwargs_from_config
from typing import Dict

__all__ = ["LayerNorm2d", "build_norm", "reset_bn", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out = x - paddle.mean(x, axis=1, keepdim=True)
        out = out / paddle.sqrt(paddle.square(out).mean(axis=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


# register normalization function here
REGISTERED_NORM_DICT: Dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None