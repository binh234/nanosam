import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant
from typing import Dict

from .utils import build_kwargs_from_config

__all__ = ["LayerNorm2D", "build_norm"]


class LayerNorm2D(nn.Layer):
    def __init__(
        self, num_features: int, eps: float = 1e-5, use_layernorm_op=False, **kwargs
    ) -> None:
        super().__init__()
        self.weight = self.create_parameter(
            shape=[num_features],
            default_initializer=Constant(1.0),
        )
        self.bias = self.create_parameter(
            shape=[num_features], default_initializer=Constant(0.0), is_bias=True
        )
        self.normalized_shape = num_features
        self.eps = eps
        self.use_layernorm_op = use_layernorm_op

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.use_layernorm_op:
            return F.layer_norm(
                x.transpose((0, 2, 3, 1)), self.normalized_shape, self.weight, self.bias, self.eps
            ).transpose((0, 3, 1, 2))
        out = x - paddle.mean(x, axis=1, keepdim=True)
        out = out / paddle.sqrt(paddle.square(out).mean(axis=1, keepdim=True) + self.eps)
        out = out * self.weight[:, None, None] + self.bias[:, None, None]
        return out

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, epsilon={self.eps}"


# register normalization function here
REGISTERED_NORM_DICT: Dict[str, type] = {
    "bn2d": nn.BatchNorm2D,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2D,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Layer or None:
    if isinstance(name, str):
        name = name.lower()
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None
