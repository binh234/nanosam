import paddle
from paddle import ParamAttr, nn
from paddle.nn.initializer import Constant, KaimingNormal
from paddle.regularizer import L2Decay
from typing import Dict, List, Optional, Tuple, Union

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

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 lr_mult=1.0):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

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
        norm="bn2d",
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
        self.norm = build_norm(
            norm,
            num_features=out_channels,
        )
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


class LightConvLayer(nn.Layer):
    """
    LightConvLayer is a combination of pw and dw layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the depth-wise convolution kernel.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
        norm (str): Normalization layer to use. Defaults to Batchnorm2D.
        act_func (str): Activation function to use. Defaults to ReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_lab=False,
        norm="bn2d",
        act_func="relu",
        **kwargs
    ):
        super().__init__()
        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            act_func=None,
            use_lab=use_lab,
        )
        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            norm=norm,
            act_func=act_func,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LearnableRepLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        num_conv_branches=1,
        lr_mult=1.0,
        lab_lr=0.1,
        act_func="relu",
    ):
        super().__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            nn.BatchNorm2D(
                num_features=in_channels,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult),
            )
            if out_channels == in_channels and stride == 1
            else None
        )

        self.conv_kxk = nn.LayerList(
            [
                ConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    lr_mult=lr_mult,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            ConvBNLayer(
                in_channels, out_channels, 1, stride, groups=groups, lr_mult=lr_mult
            )
            if kernel_size > 1
            else None
        )

        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = build_act(act_func)

    def forward(self, x):
        # for export
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2 and self.act:
                out = self.act(out)
            return out

        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2 and self.act:
            out = self.act(out)
        return out

    def re_parameterize(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)
        self.is_repped = True

    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        if not isinstance(kernel1x1, paddle.Tensor):
            return 0
        else:
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self):
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1, self.kernel_size // 2)

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias

        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(self, branch):
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class UpSampleLayer(nn.Layer):
    def __init__(
        self,
        mode="bilinear",
        size: Union[int, Tuple[int, int], List[int], None] = None,
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


class HGV2_Act_Block(nn.Layer):
    """
    HGV2_Block with custom activation, the basic unit that constitutes the HGV2_Stage.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of middle channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        layer_num (int): Number of layers in the HGV2 block. Defaults to 6.
        stride (int): Stride of the convolution. Defaults to 1.
        padding (int/str): Padding or padding type for the convolution. Defaults to 1.
        groups (int): Number of groups for the convolution. Defaults to 1.
        norm (str): Normalization layer to use. Defaults to Batchnorm2D.
        act_func (str): Activation function to use. Defaults to ReLU.
        use_lab (bool): Whether to use the LAB operation. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        layer_num=6,
        identity=False,
        light_block=True,
        norm="bn2d",
        act_func="relu",
        use_lab=False,
    ):
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        block_type = "LightConvLayer" if light_block else "ConvLayer"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    stride=1,
                    kernel_size=kernel_size,
                    norm=norm,
                    act_func=act_func,
                    use_lab=use_lab,
                )
            )
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvLayer(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm=norm,
            act_func=act_func,
            use_lab=use_lab,
        )
        self.aggregation_excitation_conv = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm=norm,
            act_func=act_func,
            use_lab=use_lab,
        )

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class LCNetV3Block(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        conv_kxk_num=4,
        lr_mult=1.0,
        lab_lr=0.1,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


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
        norm=("bn2d", "bn2d"),
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
        main: Optional[nn.Layer],
        shortcut: Optional[nn.Layer],
        post_act: str = None,
        pre_norm: Optional[nn.Layer] = None,
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
        post_input: Optional[nn.Layer],
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
    def __init__(self, op_list: List[Union[nn.Layer, None]]):
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
