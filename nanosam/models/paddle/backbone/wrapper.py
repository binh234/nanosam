import paddle.nn as nn


class ModelWrapper(nn.Layer):
    def __init__(self, model: nn.Layer, forward_func: function):
        super().__init__()
        self.model = model
        self._forward_func = forward_func

    def forward(self, x):
        return self._forward_func(self.model, x)
