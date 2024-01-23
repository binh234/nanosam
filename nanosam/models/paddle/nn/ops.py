import paddle
from paddle import ParamAttr, nn
from paddle.nn.initializer import Constant, KaimingNormal
from paddle.regularizer import L2Decay
from typing import Dict, List, Tuple

from .act import build_act
from .norm import build_norm
from .utils import get_same_padding, list_sum, resize, val2list, val2tuple

#################################################################################
#                             Basic Layers                                      #
#################################################################################


class LearnableAffineBlock(nn.Layer):
    """
    Create a learnable affine block module. This module can significantly improve accuracy on smaller models.

    Args:
        scale_value (float): The initial value of the scale parameter, default is 1.0.
        bias_value (float): The initial value of the bias parameter, default is 0.0.
        lr_mult (float): The learning rate multiplier, default is 1.0.
        lab_lr (float): The learning rate, default is 0.01.
    """

    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.01):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvLayer(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2D",
        act_func="relu",
        use_lab=False,
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.use_lab = use_lab
        self.dropout = nn.Dropout2D(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias_attr=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
        if self.act and self.use_lab:
            self.lab = LearnableAffineBlock()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class UpSampleLayer(nn.Layer):
    def __init__(
        self,
        mode="bilinear",
        size: int or Tuple[int, int] or List[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class IdentityLayer(nn.Layer):
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class FusedMBConv(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2D", "bn2D"),
        act_func=("relu6", None),
        identity=False,
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.identity = identity

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        if self.identity:
            x += identity
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Layer):
    def __init__(
        self,
        main: nn.Layer or None,
        shortcut: nn.Layer or None,
        post_act=None,
        pre_norm: nn.Layer or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Layer):
    def __init__(
        self,
        inputs: Dict[str, nn.Layer],
        merge: str,
        post_input: nn.Layer or None,
        middle: nn.Layer,
        outputs: Dict[str, nn.Layer],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.LayerList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.LayerList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = paddle.concat(feat, axis=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Layer):
    def __init__(self, op_list: List[nn.Layer or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.LayerList(valid_op_list)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
