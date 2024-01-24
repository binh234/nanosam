import numpy as np

import copy
import inspect
import paddle.distributed as dist
import random
from functools import partial
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from ppcls.data import create_operators
from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .dataset import ImageFolderDataset


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """callback function on each worker subprocess after seeding and before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1]
        num_workers (int): Number of subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in non-distributed environment, it is a constant number `0`.
        seed (int): Random seed
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(config, mode, device, use_dali=False, seed=None):
    assert mode in ["Train", "Eval"], "Dataset mode should be Train, Eval"
    assert mode in config.keys(), "{} config not in yaml".format(mode)
    # build dataset

    epochs = config.get("epochs", None)
    config_dataset = config[mode]["dataset"]
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop("name")

    dataset = eval(dataset_name)(**config_dataset)

    logger.debug("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]["sampler"]
    if config_sampler and "name" not in config_sampler:
        batch_sampler = None
        batch_size = config_sampler["batch_size"]
        drop_last = config_sampler["drop_last"]
        shuffle = config_sampler["shuffle"]
    else:
        sampler_name = config_sampler.pop("name")
        sampler_argspec = inspect.getfullargspec(eval(sampler_name).__init__).args
        if "total_epochs" in sampler_argspec:
            config_sampler.update({"total_epochs": epochs})
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)

    logger.debug("build batch_sampler({}) success...".format(batch_sampler))

    # build dataloader
    config_loader = config[mode]["loader"]
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=dist.get_rank(), seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset=dataset,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=init_fn,
    )

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader
