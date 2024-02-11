import paddle.nn as nn
from functools import partial
from typing import Dict

from .utils import build_kwargs_from_config

__all__ = ["build_act"]


# register activation function here
REGISTERED_ACT_DICT: Dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.Silu,
    "gelu": partial(nn.GELU, approximate=True),
}


def build_act(name: str, **kwargs) -> nn.Layer or None:
    if isinstance(name, str):
        name = name.lower()
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None
