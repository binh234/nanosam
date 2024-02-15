import paddle
import paddle.nn as nn
from typing import Dict

from ..nn.ops import UpSampleLayer


class ConvNeck(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        head_depth: int = 3,
        mid_channels: int = 256,
        out_channels: int = 256,
        feature_shape: int = 64,
        pos_embedding: bool = True,
        fid: str = "stage_final",
        **kwargs,
    ):
        super().__init__()
        self.fid = fid

        self.up = nn.Sequential(
            nn.Conv2D(in_channels, mid_channels, 1, padding=0), UpSampleLayer(size=(feature_shape, feature_shape))
        )

        proj_layers = []
        for _ in range(head_depth):
            proj_layers.append(nn.Conv2D(mid_channels, mid_channels, 3, padding=1))
            proj_layers.append(nn.GELU())
        proj_layers.append(nn.Conv2D(mid_channels, out_channels, 1, padding=0))
        self.proj = nn.Sequential(*proj_layers)

        if pos_embedding:
            data = 1e-5 * paddle.randn(
                (1, out_channels, feature_shape, feature_shape), dtype="float32"
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
        x = self.up(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        x = self.proj(x)
        return {"features": x}
