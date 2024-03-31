# Configuration Descriptions

------

## Introdction

The parameters in the inference configuration file(`*.yaml`) are described for you to customize or modify the hyperparameter configuration more quickly.

## Details

### Catalogue

- [Configuration Descriptions](#configuration-descriptions)
  - [Introdction](#introdction)
  - [Details](#details)
    - [Catalogue](#catalogue)
    - [1. Configuration](#1-configuration)
      - [1.1 ONNX Configuration](#11-onnx-configuration)
      - [1.2 TensorRT Configuration](#12-tensorrt-configuration)

<a name="1"></a>

### 1. Configuration

Here the configuration of [Image encoder](encoder.yaml) is used as an example to explain the each parameter in detail.

| Parameter name  | Specific meaning                        | Default value | Dtype                          |
| --------------- | --------------------------------------- | ------------- | ------------------------------ |
| name            | Model class name                        | OnnxModel     | str, `OnnxModel` or `TrtModel` |
| path            | Model path to load                      | -             | str                            |
| normalize_input | Whether to apply ImageNet normalization | True          | bool                           |

<a name="1.1"></a>

#### 1.1 ONNX Configuration

| Parameter name   | Specific meaning                                                              | Default value | Dtype              |
| ---------------- | ----------------------------------------------------------------------------- | ------------- | ------------------ |
| provider         | Sequence of providers in order of decreasing precedence                       | gpu           | str or List[str]   |
| provider_options | Sequence of options dicts corresponding to the providers listed in `provider` | null          | Dict or List[Dict] |

<a name="1.21"></a>

#### 1.2 TensorRT Configuration

To check the model's input/output names, you can either use [Netron](https://github.com/lutzroeder/netron) or [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy).

| Parameter name | Specific meaning   | Defult value | Dtype     |
| -------------- | ------------------ | ------------ | --------- |
| input_names    | Model input names  | -            | List[str] |
| output_names   | Model output names | -            | List[str] |
