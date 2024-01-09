import paddle
import paddle.nn as nn


class PaddleImageEncoder(nn.Layer):
    def __init__(self, backbone: nn.Layer, neck: nn.Layer, return_dict: bool=False) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.return_dict = return_dict

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        feed_dict = self.backbone(x)
        output = self.neck(feed_dict)
        
        if not self.return_dict and isinstance(output, dict):
            out = out["features"]
        return output
