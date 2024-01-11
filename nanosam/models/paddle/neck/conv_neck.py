from typing import Dict
import paddle
import paddle.nn as nn


class ConvNeck(nn.Layer):
    def __init__(
        self,
        in_channels: str,
        num_upsample: str,
        num_conv_layers: int = 3,
        feature_dim: int = 256,
        feature_shape: int = 64,
        neck_channels: int = 256,
        pos_embedding: bool = True,
        fid: str = "stage_final",
        **kwargs,
    ):
        super().__init__()
        self.fid = fid

        up_layers = []
        for _ in range(num_conv_layers):
            up_layers.append(nn.Conv2D(in_channels, neck_channels, 3, padding=1))
            up_layers.append(nn.GELU())
            in_channels = neck_channels

        for _ in range(num_upsample):
            up_layers.append(nn.Conv2DTranspose(neck_channels, neck_channels, 3, 2, 1, 1))
            up_layers.append(nn.GELU())

        self.up = nn.Sequential(*up_layers)

        self.proj = nn.Sequential(
            nn.Conv2D(neck_channels, neck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2D(neck_channels, feature_dim, 1, padding=0),
        )

        if pos_embedding:
            data = 1e-5 * paddle.randn(
                (1, feature_dim, feature_shape, feature_shape), dtype="float32"
            )
            pos_embedding = paddle.create_parameter(
                shape=data.shape,
                dtype=data.dtype,
                default_initializer=paddle.nn.initializer.Assign(data),
            )
            self.add_parameter("pos_embedding", pos_embedding)
        else:
            self.pos_embedding = None

    def forward(self, feed_dict: Dict[str, paddle.Tensor]):
        x = feed_dict[self.fid]
        x = self.proj(self.up(x))
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x
