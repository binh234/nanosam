import platform
from typing import Any

import onnxruntime as ort

PROVIDERS_DICT = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu_fp32": "OpenVINOExecutionProvider",
    "gpu_fp32": "OpenVINOExecutionProvider",
    "gpu_fp16": "OpenVINOExecutionProvider",
}


class OnnxModel:
    def __init__(self, path, provider="cpu", provider_options=None, **kwargs) -> None:
        if type(provider) in [tuple, list]:
            provider, provider_options = provider
        provider = provider.lower()
        provider_name = PROVIDERS_DICT.get(provider, provider)

        if platform.system() == "Windows" and provider_name == "OpenVINOExecutionProvider":
            # noqa
            from openvino import utils

            utils.add_openvino_libs_to_path()

            if provider in ["cpu_fp32", "gpu_fp32", "gpu_fp16"] and provider_options is None:
                provider_options = {"device_type": provider.upper()}

        if isinstance(provider_options, dict):
            provider_options = [provider_options]
        self.session = ort.InferenceSession(
            path, providers=[provider_name], provider_options=provider_options, **kwargs
        )
        self.input_dict = {}
        self.input_names = []

        for node in self.session.get_inputs():
            self.input_dict[node.name] = None
            self.input_names.append(node.name)

    def forward(self, *args: Any) -> Any:
        assert len(args) == len(
            self.input_names
        ), "Number of arguments must match number of model's input"
        for name, data in zip(self.input_names, args):
            self.input_dict[name] = data
        return self.session.run(None, self.input_dict)

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def get_modelmeta(self):
        return self.session.get_modelmeta()

    def set_providers(self, providers, provider_options=None):
        self.session.set_providers(providers, provider_options)
