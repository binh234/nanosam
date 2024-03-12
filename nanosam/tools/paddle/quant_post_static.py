from __future__ import absolute_import, division, print_function
import numpy as np

import os
import paddle
import paddleslim
from ppcls.utils import config as conf
from ppcls.utils.logger import init_logger

from .loader import build_dataloader


def main():
    args = conf.parse_args()
    config = conf.get_config(args.config, overrides=args.override, show=False)

    assert os.path.exists(
        os.path.join(config["Global"]["save_inference_dir"], "inference.pdmodel")
    ) and os.path.exists(
        os.path.join(config["Global"]["save_inference_dir"], "inference.pdiparams")
    )
    if "Query" in config["DataLoader"]["Eval"]:
        config["DataLoader"]["Eval"] = config["DataLoader"]["Eval"]["Query"]
    config["DataLoader"]["Eval"]["sampler"]["batch_size"] = 1
    config["DataLoader"]["Eval"]["loader"]["num_workers"] = 0

    init_logger()
    device = paddle.set_device("cpu")
    train_dataloader = build_dataloader(config["DataLoader"], "Eval", device, False)

    def sample_generator(loader):
        def __reader__():
            for indx, data in enumerate(loader):
                images = np.array(data[0])
                yield images

        return __reader__

    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir=config["Global"]["save_inference_dir"],
        model_filename="inference.pdmodel",
        params_filename="inference.pdiparams",
        quantize_model_path=os.path.join(
            config["Global"]["save_inference_dir"], "quant_post_static_model"
        ),
        sample_generator=sample_generator(train_dataloader),
        batch_size=config["DataLoader"]["Eval"]["sampler"]["batch_size"],
        batch_nums=10,
    )


if __name__ == "__main__":
    main()
