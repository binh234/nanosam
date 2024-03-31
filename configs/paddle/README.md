# Configuration Descriptions

------

## Introdction

The parameters in the configuration file(`*.yaml`) are described for you to customize or modify the hyperparameter configuration more quickly.

## Details

### Catalogue

- [Configuration Descriptions](#configuration-descriptions)
  - [Introdction](#introdction)
  - [Details](#details)
    - [Catalogue](#catalogue)
    - [1. Configuration](#1-configuration)
      - [1.1 Global Configuration](#11-global-configuration)
      - [1.2 Architecture](#12-architecture)
      - [1.2.1 Teacher](#121-teacher)
      - [1.3 Loss function](#13-loss-function)
      - [1.4 Optimizer](#14-optimizer)
      - [1.5 Data reading module(DataLoader)](#15-data-reading-moduledataloader)
        - [1.5.1 dataset](#151-dataset)
        - [1.5.2 sampler](#152-sampler)
        - [1.5.3 loader](#153-loader)

<a name="1"></a>

### 1. Configuration

Here the configuration of [Sam_PPHGV2_B4](sam_pphgv2_b4_nonorm.yaml) is used as an example to explain the each parameter in detail.

<a name="1.1"></a>

#### 1.1 Global Configuration

| Parameter name       | Specific meaning                                           | Default value | Dtype |
| -------------------- | ---------------------------------------------------------- | ------------- | ----- |
| checkpoints          | Breakpoint model path for resuming training                | null          | str   |
| pretrained_model     | Pre-trained model path                                     | null          | str   |
| device               | Training device                                            | gpu           | str   |
| output_dir           | Save model path                                            | "./output/"   | str   |
| save_interval        | How many epochs to save the model at each interval         | 1             | int   |
| eval_during_train    | Whether to evaluate at training                            | True          | bool  |
| eval_interval        | How many epochs to evaluate at each interval               | 1             | int   |
| epochs               | Total number of epochs in training                         | 8             | int   |
| print_batch_step     | How many mini-batches to print out at each interval        | 500           | int   |
| use_visualdl         | Whether to visualize the training process with visualdl    | False         | bool  |
| save_inference_dir   | Inference model save path                                  | "./inference" | str   |
| student_size         | Student model input image size                             | 512           | int   |
| teacher_size         | Teacher model input image size                             | 512           | int   |
| export_dynamic_batch | Dynamic batch export                                       | False         | bool  |
| image_shape          | Image size for export                                      | [3，512, 512] | list  |
| print_model          | Whether to print the model architecture, use for debugging | [3，512, 512] | list  |

**Note**：The http address of pre-trained model can be filled in the `pretrained_model`

<a name="1.2"></a>

#### 1.2 Architecture

| Parameter name   | Specific meaning                                                      | Default value    | Dtype                                              |
| ---------------- | --------------------------------------------------------------------- | ---------------- | -------------------------------------------------- |
| name             | Model Arch name                                                       | Sam_PPHGNetV2_B4 | str                                                |
| pretrained       | Pre-trained model                                                     | False            | bool                                               |
| use_ssld         | Whether to use SSLD pre-train model                                   | False            | bool                                               |
| middle_op        | Middle block to be used in SAM head                                   | hgv2             | str, in `fmb`, `hgv2`, `hgv2_act`, `repdw`, `lcv3` |
| head_depth       | SAM head depth                                                        | 6                | int                                                |
| num_block        | Number of middle blocks in SAM head                                   | 2                | int                                                |
| expand_ratio     | Expand ratio of middle layers                                         | 1                | float                                              |
| use_last_norm    | Whether to use layernorm2d as final layer                             | True             | bool                                               |
| use_layernorm_op | Whether to use layernorm op, use with ONNX opset>17 for better export | False            | bool                                               |

**Note**: `pretrained` can be set to True or False, so does the path of the weights. In addition, the pretrained is disabled when Global.pretrained_model is also set to the corresponding path.

<a name="1.2.1"></a>

#### 1.2.1 Teacher

Teacher model used for distillation

| Parameter name | Specific meaning | Default value | Dtype |
| -------------- | ---------------- | ------------- | ----- |
| name           | Model Arch name  | TrtModel      | str   |
| path           | Model path       | -             | str   |

<a name="1.3"></a>

#### 1.3 Loss function

| Parameter name         | Specific meaning                               | Default value | Dtype                                                     |
| ---------------------- | ---------------------------------------------- | ------------- | --------------------------------------------------------- |
| DistanceLoss           | Distillation loss function                     | ——            | ——                                                        |
| DistanceLoss.mode      | The loss computation mode                      | sqrt_l2       | str, in `l1`, `smooth_l1` (huber), `l2`, `sqrt_l2` (rmse) |
| DistanceLoss.weight    | The weight of DistanceLoss in the whole loss   | 1.0           | float                                                     |
| DistanceLoss.reduction | Specifies the reduction to apply to the output | 'size_sum     | str, in `none`, `mean`, `sum`, `size_sum`                 |

<a name="1.4"></a>

#### 1.4 Optimizer

| Parameter name                | Specific meaning                             | Default value | Dtype                               |
| ----------------------------- | -------------------------------------------- | ------------- | ----------------------------------- |
| name                          | optimizer method name                        | "AdamW"       | Other optimizer including "RmsProp" |
| one_dim_param_no_weight_decay | Whether to not apply regularization for bias | True          | bool                                |
| weight_decay                  | regularization factor                        | 0.0001        | float                               |
| lr.name                       | method of dropping learning rate             | "Cosine"      | str                                 |
| lr.learning_rate              | initial value of learning rate               | 0.1           | float                               |
| lr.warmup_epoch               | warmup rounds                                | 0             | int，such as 5                      |

Referring to [learning_rate.py](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/optimizer/learning_rate.py) for adding method and parameters.

<a name="1.5"></a>

#### 1.5 Data reading module(DataLoader)

<a name="1.5.1"></a>

##### 1.5.1 dataset

| Parameter name      | Specific meaning                       | Default value      | Dtype |
| ------------------- | -------------------------------------- | ------------------ | ----- |
| name                | The name of the class to read the data | ImageFolderDataset | str   |
| image_root          | The path where the dataset is stored   | /workspace/sa1b    | str   |
| transform_ops       | data preprocessing for single image    | ——                 | ——    |
| batch_transform_ops | Data preprocessing for batch images    | ——                 | ——    |

The parameter meaning of transform_ops:

| Function name  | Parameter name | Specific meaning                             |
| -------------- | -------------- | -------------------------------------------- |
| RandCropImage  |                | Random crop                                  |
| RandFlipImage  |                | Random flip                                  |
| SamResizeImage | resize_long    | Resize image's longest size to `resize_long` |
| NormalizeImage | scale          | Normalize scale value                        |
|                | mean           | Normalize mean value                         |
|                | std            | Normalized variance                          |
|                | order          | Normalize order, `chw` or `hwc`              |
| SamPad         | size           | Pad image to specific size                   |

<a name="1.5.2"></a>

##### 1.5.2 sampler

| Parameter name | Specific meaning                                             | Default value           | Dtype                                              |
| -------------- | ------------------------------------------------------------ | ----------------------- | -------------------------------------------------- |
| name           | sampler type                                                 | DistributedBatchSampler | DistributedRandomIdentitySampler and other Sampler |
| batch_size     | batch size                                                   | 8                       | int                                                |
| drop_last      | Whether to drop the last data that does reach the batch-size | True                    | bool                                               |
| shuffle        | whether to shuffle the data                                  | True                    | bool                                               |

<a name="1.5.3"></a>

##### 1.5.3 loader

| Parameter name    | Specific meaning             | Default meaning | Optional meaning |
| ----------------- | ---------------------------- | --------------- | ---------------- |
| num_workers       | Number of data read threads  | 4               | int              |
| use_shared_memory | Whether to use shared memory | True            | bool             |
