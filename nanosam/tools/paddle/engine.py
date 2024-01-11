from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.arch import apply_to_static
from ppcls.loss import build_loss
from ppcls.optimizer import build_optimizer
from ppcls.utils.amp import AutoCast, build_scaler
from ppcls.utils.checkpointer import Checkpointer
from ppcls.utils.ema import ExponentialMovingAverage
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils.save_load import init_model

from ppcls.engine.train.utils import type_name

from nanosam.models.paddle import build_model
from nanosam.utils.onnx_model import OnnxModel
from .loader import build_dataloader
from .train_utils import train_epoch, eval_epoch
from .trt_model import TrtModel


class Engine(object):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode = mode
        self.config = config

        # set seed
        seed = self.config["Global"].get("seed", False)
        if seed or seed == 0:
            assert isinstance(seed, int), "The 'seed' must be a integer!"
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # init logger
        self.output_dir = self.config["Global"]["output_dir"]
        log_file = os.path.join(self.output_dir, self.config["Arch"]["name"], f"{mode}.log")
        init_logger(log_file=log_file)
        if self.mode == "train" or self.config["Global"].get("print_config", False):
            print_config(config)

        # init checkpointer
        ckpt_config = self.config.get("Checkpointer", {})
        if "checkpoint_dir" not in ckpt_config:
            ckpt_config["checkpoint_dir"] = os.path.join(
                self.output_dir, self.config["Arch"]["name"]
            )
        self.checkpointer = Checkpointer(**ckpt_config)

        # init train_func and eval_func
        self.train_epoch_func = train_epoch
        self.eval_func = eval_epoch

        self.use_dali = self.config["Global"].get("use_dali", False)

        # for visualdl
        self.vdl_writer = None
        if self.config["Global"]["use_visualdl"] and mode == "train" and dist.get_rank() == 0:
            vdl_writer_path = self.output_dir
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.config["Global"]["device"] in [
            "cpu",
            "gpu",
            "xpu",
            "npu",
            "mlu",
            "ascend",
            "intel_gpu",
            "mps",
        ]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info("train with paddle {} and device {}".format(paddle.__version__, self.device))

        # gradient accumulation
        self.update_freq = self.config["Global"].get("update_freq", 1)
        self.config["DataLoader"].update({"epochs": self.config["Global"]["epochs"]})

        # build dataloader
        if self.mode == "train":
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali
            )

            self.iter_per_epoch = (
                len(self.train_dataloader) - 1
                if platform.system() == "Windows"
                else len(self.train_dataloader)
            )
            if self.config["Global"].get("iter_per_epoch", None):
                # set max iteration per epoch mannualy, when training by iteration(s), such as XBM, FixMatch.
                self.iter_per_epoch = self.config["Global"].get("iter_per_epoch")
            if self.iter_per_epoch < self.update_freq:
                logger.warning(
                    "The arg Global.update_freq greater than iter_per_epoch and has been set to 1. This may be caused by too few of batches."
                )
                self.update_freq = 1
            self.iter_per_epoch = self.iter_per_epoch // self.update_freq * self.update_freq

        if self.mode == "eval" or (
            self.mode == "train" and self.config["Global"]["eval_during_train"]
        ):
            self.eval_dataloader = build_dataloader(
                self.config["DataLoader"], "Eval", self.device, self.use_dali
            )

        # build loss
        if self.mode == "train":
            label_loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = build_loss(label_loss_info)
        if self.mode == "eval" or (
            self.mode == "train" and self.config["Global"]["eval_during_train"]
        ):
            loss_config = self.config.get("Loss", None)
            if loss_config is not None:
                loss_config = loss_config.get("Eval")
                if loss_config is not None:
                    self.eval_loss_func = build_loss(loss_config)
                else:
                    self.eval_loss_func = None
            else:
                self.eval_loss_func = None

        # build metric
        self.train_metric_func = None
        self.eval_metric_func = None

        # build model
        self.model = build_model(self.config["Arch"])
        # set @to_static for benchmark, skip this by default.
        apply_to_static(self.config, self.model)
        if self.config["Global"].get("print_model", False):
            print("Model architecture")
            print(self.model)
        
        # SAM Teacher
        self.train_batch_size = config["DataLoader"]["Train"]["sampler"]["batch_size"]
        if self.mode in ["train", "eval"]:
            teacher_model_config = config["Teacher"]
            teacher_model_type = teacher_model_config.pop("name", "TrtModel")
            self.teacher_model = eval(teacher_model_type)(**teacher_model_config)

        # load_pretrain
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    [self.model, getattr(self, "train_loss_func", None)],
                    self.config["Global"]["pretrained_model"],
                )
            else:
                print("Loading ", self.config["Global"]["pretrained_model"])
                load_dygraph_pretrain(
                    [self.model, getattr(self, "train_loss_func", None)],
                    self.config["Global"]["pretrained_model"],
                )

        # build optimizer
        if self.mode == "train":
            self.optimizer, self.lr_sch = build_optimizer(
                self.config["Optimizer"],
                self.config["Global"]["epochs"],
                self.iter_per_epoch // self.update_freq,
                [self.model, self.train_loss_func],
            )
        # amp
        self._init_amp()

        # build EMA model
        self.ema = "EMA" in self.config and self.mode == "train"
        if self.ema:
            self.model_ema = ExponentialMovingAverage(
                self.model, self.config["EMA"].get("decay", 0.9999)
            )

        # check the gpu num
        world_size = dist.get_world_size()
        self.config["Global"]["distributed"] = world_size != 1

        # for distributed
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)
            if self.mode == "train" and len(self.train_loss_func.parameters()) > 0:
                self.train_loss_func = paddle.DataParallel(self.train_loss_func)

            # set different seed in different GPU manually in distributed environment
            if seed is None:
                logger.warning(
                    "The random seed cannot be None in a distributed environment. Global.seed has been set to 42 by default"
                )
                self.config["Global"]["seed"] = seed = 42
            logger.info(
                f"Set random seed to ({int(seed)} + $PADDLE_TRAINER_ID) for different trainer"
            )
            paddle.seed(int(seed) + dist.get_rank())
            np.random.seed(int(seed) + dist.get_rank())
            random.seed(int(seed) + dist.get_rank())

    def train(self):
        assert self.mode == "train"
        print_batch_step = self.config["Global"]["print_batch_step"]
        save_interval = self.config["Global"]["save_interval"]
        best_metric = {
            "metric": -1.0,
            "epoch": 0,
        }
        ema_module = None
        if self.ema:
            best_metric_ema = 0.0
            ema_module = self.model_ema.module
        # key:
        # val: metrics list word
        self.output_info = dict()
        self.time_info = {
            "batch_cost": AverageMeter("batch_cost", ".5f", postfix=" s,"),
            "reader_cost": AverageMeter("reader_cost", ".5f", postfix=" s,"),
        }
        # global iter counter
        self.global_step = 0

        if self.config.Global.checkpoints is not None:
            metric_info = init_model(
                self.config.Global, self.model, self.optimizer, self.train_loss_func, ema_module
            )
            if metric_info is not None:
                best_metric.update(metric_info)
            if hasattr(self.train_dataloader.batch_sampler, "set_epoch"):
                self.train_dataloader.batch_sampler.set_epoch(best_metric["epoch"])

        for epoch_id in range(best_metric["epoch"] + 1, self.config["Global"]["epochs"] + 1):
            loss = 0.0
            # for one epoch train
            self.train_epoch_func(self, epoch_id, print_batch_step)

            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join([self.output_info[key].avg_info for key in self.output_info])
            logger.info(
                "[Train][Epoch {}/{}][Avg]{}".format(
                    epoch_id, self.config["Global"]["epochs"], metric_msg
                )
            )
            self.output_info.clear()

            # eval model and save model if possible
            start_eval_epoch = self.config["Global"].get("start_eval_epoch", 0) - 1
            if (
                self.eval_func is not None
                and self.config["Global"]["eval_during_train"]
                and epoch_id % self.config["Global"]["eval_interval"] == 0
                and epoch_id > start_eval_epoch
            ):
                loss = self.eval(epoch_id)

                # step lr (by epoch) according to given metric, such as loss
                for i in range(len(self.lr_sch)):
                    if (
                        getattr(self.lr_sch[i], "by_epoch", False)
                        and type_name(self.lr_sch[i]) == "ReduceOnPlateau"
                    ):
                        self.lr_sch[i].step(loss)

                if loss < best_metric["metric"]:
                    best_metric["metric"] = loss
                    best_metric["epoch"] = epoch_id
                    self.checkpointer.save_checkpoint(
                        self.model,
                        self.optimizer,
                        best_metric,
                        ema=ema_module,
                        prefix="best_model",
                        loss=self.train_loss_func,
                        save_student_model=True,
                    )
                logger.info(
                    "[Eval][Epoch {}][best metric: {}]".format(epoch_id, best_metric["metric"])
                )
                logger.scaler(name="eval_loss", value=loss, step=epoch_id, writer=self.vdl_writer)

                self.model.train()

                if self.ema:
                    ori_model, self.model = self.model, ema_module
                    acc_ema = self.eval(epoch_id, is_ema=True)
                    self.model = ori_model
                    ema_module.eval()

                    if acc_ema > best_metric_ema:
                        best_metric_ema = acc_ema
                        self.checkpointer.save_checkpoint(
                            ema_module,
                            None,
                            {"metric": acc_ema, "epoch": epoch_id},
                            ema=None,
                            prefix="best_model.ema",
                            loss=self.train_loss_func,
                        )
                    logger.info(
                        "[Eval][Epoch {}][best metric ema: {}]".format(epoch_id, best_metric_ema)
                    )
                    logger.scaler(
                        name="eval_loss_ema", value=acc_ema, step=epoch_id, writer=self.vdl_writer
                    )

            # save model
            if save_interval > 0 and epoch_id % save_interval == 0:
                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    {"metric": loss, "epoch": epoch_id},
                    ema=ema_module,
                    epoch_id=epoch_id,
                    loss=self.train_loss_func,
                )
            # save the latest model
            self.checkpointer.save_checkpoint(
                self.model,
                self.optimizer,
                {"metric": loss, "epoch": epoch_id},
                ema=ema_module,
                prefix="latest",
                loss=self.train_loss_func,
            )

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self, epoch_id=0, is_ema=False):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_func(self, epoch_id, is_ema=is_ema)
        self.model.train()
        return eval_result

    def export(self):
        assert self.mode == "export"

        model = ExportModel(self.config["Arch"], self.model)
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    model.base_model, self.config["Global"]["pretrained_model"]
                )
            else:
                load_dygraph_pretrain(model.base_model, self.config["Global"]["pretrained_model"])

        model.eval()

        # for re-parameterization nets
        for layer in self.model.sublayers():
            if hasattr(layer, "re_parameterize") and not getattr(layer, "is_repped"):
                layer.re_parameterize()

        save_path = os.path.join(self.config["Global"]["save_inference_dir"], "inference")

        export_dynamic_batch = self.config["Global"].get("export_dynamic_batch", False)
        logger.info(f"Dynamic batch: {export_dynamic_batch}")

        if export_dynamic_batch:
            shape = [None] + self.config["Global"]["image_shape"]
        else:
            export_batch_size = self.config["Global"].get("export_batch_size", 1)
            shape = [export_batch_size] + self.config["Global"]["image_shape"]

        logger.info(f"Export input shape: {shape}")
        model = paddle.jit.to_static(
            model, input_spec=[paddle.static.InputSpec(shape=shape, dtype="float32")]
        )
        if hasattr(model.base_model, "quanter") and model.base_model.quanter is not None:
            model.base_model.quanter.save_quantized_model(model, save_path + "_int8")
        else:
            paddle.jit.save(model, save_path)
        if self.config["Global"].get("export_for_fd", False):
            src_path = self.config["Global"]["infer_config_path"]
            dst_path = os.path.join(self.config["Global"]["save_inference_dir"], "inference.yml")
            shutil.copy(src_path, dst_path)
        logger.info(
            f"Export succeeded! The inference model exported has been saved in \"{self.config['Global']['save_inference_dir']}\"."
        )

    def _init_amp(self):
        if self.mode == "export":
            return

        amp_config = self.config.get("AMP", None)
        use_amp = True if amp_config and amp_config.get("use_amp", True) else False

        if not use_amp:
            self.auto_cast = AutoCast(use_amp)
            self.scaler = build_scaler(use_amp)
        else:
            AMP_RELATED_FLAGS_SETTING = {
                "FLAGS_max_inplace_grad_add": 8,
            }
            if paddle.is_compiled_with_cuda():
                AMP_RELATED_FLAGS_SETTING.update({"FLAGS_cudnn_batchnorm_spatial_persistent": 1})
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)

            use_promote = amp_config.get("use_promote", False)
            amp_level = amp_config.get("level", "O1")
            if amp_level not in ["O1", "O2"]:
                msg = "[Parameter Error]: The optimize level of AMP only support 'O1' and 'O2'. The level has been set 'O1'."
                logger.warning(msg)
                amp_level = amp_config["level"] = "O1"

            amp_eval = self.config["AMP"].get("use_fp16_test", False)
            # TODO(gaotingquan): Paddle not yet support FP32 evaluation when training with AMPO2
            if (
                self.mode == "train"
                and self.config["Global"].get("eval_during_train", True)
                and amp_level == "O2"
                and amp_eval == False
            ):
                msg = "PaddlePaddle only support FP16 evaluation when training with AMP O2 now. "
                logger.warning(msg)
                self.config["AMP"]["use_fp16_test"] = True
                amp_eval = True

            self.auto_cast = AutoCast(
                use_amp, amp_level=amp_level, use_promote=use_promote, amp_eval=amp_eval
            )

            scale_loss = amp_config.get("scale_loss", 1.0)
            use_dynamic_loss_scaling = amp_config.get("use_dynamic_loss_scaling", False)
            self.scaler = build_scaler(
                use_amp, scale_loss=scale_loss, use_dynamic_loss_scaling=use_dynamic_loss_scaling
            )

            if self.mode == "train":
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=amp_level,
                    save_dtype="float32",
                )
            elif amp_eval:
                self.model = paddle.amp.decorate(
                    models=self.model, level=amp_level, save_dtype="float32"
                )

            if self.mode == "train" and len(self.train_loss_func.parameters()) > 0:
                self.train_loss_func = paddle.amp.decorate(
                    models=self.train_loss_func, level=amp_level, save_dtype="float32"
                )


class ExportModel(nn.Layer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, model):
        super().__init__()
        self.base_model = model

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.base_model(x)
        if isinstance(x, list):
            x = x[0]

        return x
