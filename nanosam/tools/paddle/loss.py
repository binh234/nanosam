import copy
import paddle
import paddle.nn as nn


class DistanceLoss(nn.Layer):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", reduction="size_sum", **kwargs):
        super().__init__()
        assert mode in ["l1", "l2", "sqrt_l2", "smooth_l1"]
        assert reduction in ["mean", "sum", "size_sum"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(reduction="none", **kwargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(reduction="none", **kwargs)
        elif mode in ["l2", "sqrt_l2"]:
            self.loss_func = nn.MSELoss(reduction="none", **kwargs)
        self.mode = mode
        self.reduction = reduction

    def forward(self, x, y):
        loss = self.loss_func(x, y)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "mean":
            loss = loss.sum()
        elif self.reduction == "size_sum":
            loss = loss.sum(1).mean()

        if self.mode == "sqrt_l2":
            loss = loss.sqrt()

        return {"loss_{}".format(self.mode): loss}


class CombinedLoss(nn.Layer):
    def __init__(self, config_list):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(config_list, list), "operator config should be a list"
        for config in config_list:
            assert isinstance(config, dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys()
            )
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))
            self.loss_func = nn.LayerList(self.loss_func)

    def __call__(self, input, batch):
        loss_dict = {}
        # just for accelerate classification traing speed
        if len(self.loss_func) == 1:
            loss = self.loss_func[0](input, batch)
            loss_dict.update(loss)
            loss_dict["loss"] = list(loss.values())[0]
        else:
            for idx, loss_func in enumerate(self.loss_func):
                loss = loss_func(input, batch)
                weight = self.loss_weight[idx]
                loss = {key: loss[key] * weight for key in loss}
                loss_dict.update(loss)
            loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict


def build_loss(config):
    if config is None:
        return None
    module_class = CombinedLoss(copy.deepcopy(config))
    return module_class
