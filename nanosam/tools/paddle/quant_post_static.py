from __future__ import absolute_import, division, print_function
import numpy as np

from nanosam.tools.paddle.dataset import ImageFolderDataset, SamImageFolderDataset

import copy
import os
import paddle
import paddleslim
from paddle.io import DataLoader, Dataset
from ppcls.utils import config as conf
from ppcls.utils.logger import init_logger


class DatasetWrapper(Dataset):
    def __init__(self, dataset, input_name="image"):
        self.dataset = dataset
        self.input_name = input_name

    def try_getitem(self, index):
        return {self.input_name: self.dataset[index][0]}

    def __getitem__(self, index):
        try:
            return self.try_getitem(index)
        except Exception as ex:
            print("Exception occured with msg: {}".format(ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.dataset)


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

    init_logger()
    device = paddle.set_device(config["Global"]["device"])

    config_dataset = config["DataLoader"]["Eval"]["dataset"]
    batch_size = 1
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop("name")
    dataset = eval(dataset_name)(**config_dataset)
    dataset = DatasetWrapper(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        places=device,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )

    paddle.enable_static()
    exe = paddle.static.Executor(device)
    paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir=config["Global"]["save_inference_dir"],
        model_filename="inference.pdmodel",
        params_filename="inference.pdiparams",
        quantize_model_path=os.path.join(
            config["Global"]["save_inference_dir"], "quant_post_static_model"
        ),
        data_loader=dataloader,
        batch_size=batch_size,
        batch_nums=10,
    )


if __name__ == "__main__":
    main()
