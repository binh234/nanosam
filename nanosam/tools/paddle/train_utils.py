# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt

import os
import paddle
import paddle.nn.functional as F
import platform
import time
from ppcls.engine.train.utils import log_info, type_name, update_loss
from ppcls.utils import logger, profiler
from ppcls.utils.misc import AverageMeter


def train_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)

    for iter_id in range(engine.iter_per_epoch):
        # fetch data batch from dataloader
        try:
            batch = next(engine.train_dataloader_iter)
        except Exception:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            batch = next(engine.train_dataloader_iter)

        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        batch_size = batch[0].shape[0]
        if batch_size != engine.train_batch_size:
            continue

        engine.global_step += 1

        # image input
        with engine.auto_cast(is_eval=False):
            inp_np = batch[0].numpy()
            targets = engine.teacher_model(inp_np)[0]
            targets = paddle.to_tensor(targets)._to(engine.device)
            if batch[0].shape[-1] != engine.model.img_size:
                batch[0] = F.interpolate(batch[0], (engine.model.img_size, engine.model.img_size))
            out = engine.model(batch[0])
            loss_dict = engine.train_loss_func(out, targets)

        # loss
        loss = loss_dict["loss"] / engine.update_freq

        # backward & step opt
        scaled = engine.scaler.scale(loss)
        scaled.backward()
        if (iter_id + 1) % engine.update_freq == 0:
            for i in range(len(engine.optimizer)):
                # optimizer.step() with auto amp
                engine.scaler.step(engine.optimizer[i])
                engine.scaler.update()

        if (iter_id + 1) % engine.update_freq == 0:
            # clear grad
            for i in range(len(engine.optimizer)):
                engine.optimizer[i].clear_grad()
            # step lr(by step)
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    engine.lr_sch[i].step()
            # update ema
            if engine.ema:
                engine.model_ema.update(engine.model)

        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()

    # step lr(by epoch)
    for i in range(len(engine.lr_sch)):
        if (
            getattr(engine.lr_sch[i], "by_epoch", False)
            and type_name(engine.lr_sch[i]) != "ReduceOnPlateau"
        ):
            engine.lr_sch[i].step()

    image_dir = os.path.join(engine.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(targets[0, 0].detach().cpu().numpy())
    plt.subplot(122)
    plt.imshow(out[0, 0].detach().cpu().numpy())
    plt.savefig(os.path.join(image_dir, f"epoch_{epoch_id}.png"))
    plt.show()


def eval_epoch(engine, epoch_id, is_ema=False):
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter("batch_cost", ".5f", postfix=" s,"),
        "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]
    tic = time.time()

    total_samples = (
        len(engine.eval_dataloader.dataset) if not engine.use_dali else engine.eval_dataloader.size
    )
    max_iter = (
        len(engine.eval_dataloader) - 1
        if platform.system() == "Windows"
        else len(engine.eval_dataloader)
    )

    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]

        if batch_size != engine.train_batch_size:
            continue

        # image input
        with engine.auto_cast(is_eval=True):
            inp_np = batch[0].numpy()
            targets = engine.teacher_model(inp_np)[0]
            targets = paddle.to_tensor(targets)._to(engine.device)
            if batch[0].shape[-1] != engine.model.img_size:
                batch[0] = F.interpolate(batch[0], (engine.model.img_size, engine.model.img_size))
            out = engine.model(batch[0])
            loss_dict = engine.eval_loss_func(out, targets)

            # Update loss
            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, "7.5f")
                output_info[key].update(float(loss_dict[key]), batch_size)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join(
                ["{}: {:.5f}".format(key, time_info[key].avg) for key in time_info]
            )
            ips_msg = "ips: {:.5f} images/sec".format(batch_size / time_info["batch_cost"].avg)
            metric_msg = ", ".join(
                ["{}: {:.5f}".format(key, output_info[key].val) for key in output_info]
            )
            logger.info(
                "[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                    epoch_id, iter_id, len(engine.eval_dataloader), metric_msg, time_msg, ips_msg
                )
            )

        tic = time.time()

    metric_msg = ", ".join(["{}: {:.5f}".format(key, output_info[key].avg) for key in output_info])
    metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    eval_loss = sum([output_info[key].avg for key in output_info]) / len(output_info)
    return eval_loss
