import numpy as np

from nanosam.mod import lazy_import

torch = lazy_import("torch")
trt = lazy_import("tensorrt")


def trt_version():
    return trt.__version__


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= "7.0" and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


class TrtModel:
    def __init__(self, path, input_names, output_names, **kwargs):
        """Initialize TensorRT plugins, engine and conetxt."""
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(path, "rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names

    def get_input_shapes(self, index: int = None):
        if index is None:
            return [
                tuple(self.engine.get_binding_shape(input_name)) for input_name in self.input_names
            ]
        else:
            return tuple(self.engine.get_binding_shape(self.input_names[index]))

    def get_output_shapes(self, index: int = None):
        if index is None:
            return [
                tuple(self.engine.get_binding_shape(output_name))
                for output_name in self.output_names
            ]
        else:
            return tuple(self.engine.get_binding_shape(self.output_names[index]))

    def infer(self, *inputs):
        """Run inference on TensorRT engine."""
        inputs = [torch.from_numpy(np.ascontiguousarray(inp)).cuda() for inp in inputs]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        outputs = [o.detach().cpu().numpy() for o in outputs]
        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def __call__(self, *args):
        out = self.infer(*args)

        return out
