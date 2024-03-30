from nanosam.mod import lazy_import

import collections
import collections.abc
import platform
import warnings
from typing import Any

ort = lazy_import("onnxruntime")

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


def get_available_providers():
    available_providers = set(ort.get_available_providers())
    for key, provider_name in PROVIDERS_DICT.items():
        if provider_name in available_providers:
            available_providers.add(key)
    return list(available_providers)


def check_and_normalize_provider_args(providers, provider_options, available_provider_names):
    """
    Validates the 'providers' and 'provider_options' arguments and returns a
        normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if name not in available_provider_names:
            warnings.warn(
                "Specified provider '{}' is not in available provider names."
                "Available providers: '{}'".format(name, ", ".join(available_provider_names))
            )

        if name in ["cpu_fp32", "gpu_fp32", "gpu_fp16"] and options is None:
            options = {"device_type": provider.upper()}

        name = PROVIDERS_DICT.get(name, name)
        if name in provider_name_to_options:
            warnings.warn("Duplicate provider '{}' encountered, ignoring.".format(name))
            return

        if platform.system() == "Windows" and name == "OpenVINOExecutionProvider":
            # noqa
            from openvino import utils

            utils.add_openvino_libs_to_path()

        normalized_options = {str(key): str(value) for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if isinstance(providers, str):
        providers = [providers]
    elif not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if isinstance(provider_options, dict):
            provider_options = [provider_options]
        elif not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError(
                "'providers' and 'provider_options' should be the same length if both are given."
            )

        if not all([isinstance(provider, str) for provider in providers]):
            raise ValueError(
                "Only string values for 'providers' are supported if 'provider_options' is given."
            )

        if not all(
            [isinstance(options_for_provider, dict) for options_for_provider in provider_options]
        ):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, dict())
            elif (
                isinstance(provider, tuple)
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError(
                    "'providers' values must be either strings or (string, dict) tuples."
                )

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


class OnnxModel:
    def __init__(self, path, provider="cpu", provider_options=None, **kwargs) -> None:
        available_providers = get_available_providers()
        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            provider, provider_options, available_providers
        )
        self.session = ort.InferenceSession(
            path, providers=providers, provider_options=provider_options, **kwargs
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
        outputs = self.session.run(None, self.input_dict)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def get_inputs(self):
        return self.session.get_inputs()

    def get_input_shapes(self, index: int = None):
        inputs = self.get_inputs()
        if index is None:
            return [inp.shape for inp in inputs]
        else:
            return inputs[index].shape

    def get_outputs(self):
        return self.session.get_outputs()

    def get_output_shapes(self, index: int = None):
        outputs = self.get_outputs()
        if index is None:
            return [out.shape for out in outputs]
        else:
            return outputs[index].shape

    def get_modelmeta(self):
        return self.session.get_modelmeta()

    def set_providers(self, providers, provider_options=None):
        self.session.set_providers(providers, provider_options)
