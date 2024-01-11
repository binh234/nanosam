import numpy as np
import paddle
import tensorrt as trt


class TrtModel:
    def __init__(self, path, device, **kwargs):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.device = device
        self.load_engine(path)

    def load_engine(self, path):
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(path, "rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.get_inference_info(self.engine)

    def get_inference_info(self, engine):
        self.context = engine.create_execution_context()
        self.stream = paddle.device.cuda.current_stream()
        self.input_shapes = []
        self.output_shapes = []
        self.binding_outputs = []
        self.outputs = []

        for binding in engine:
            shape = engine.get_tensor_shape(binding)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))

            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_shapes.append(shape)
            else:
                tensor = paddle.to_tensor(np.empty(shape, dtype=np.dtype(dtype)), place=self.device)
                self.binding_outputs.append(tensor.data_ptr())
                self.outputs.append(tensor)
                self.output_shapes.append(shape)

    def infer(self, *args):
        """Run inference on TensorRT engine."""
        assert len(args) == len(
            self.input_shapes
        ), "Number of arguments must match number of model's input"

        binding_outputs = [inp.data_ptr() for inp in args] + self.binding_outputs
        self.context.execute_async_v2(
            bindings=binding_outputs, stream_handle=self.stream.cuda_stream
        )
        self.stream.synchronize()

        return self.outputs  # paddle tensors on GPU

    def __call__(self, *args):
        out = self.infer(*args)

        return out
