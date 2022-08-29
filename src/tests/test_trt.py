import sys
# Hack to allow loading the pickled yolov7 model
sys.path.insert(0, "src/train/yolov7")

import torch
import tensorrt as trt
import unittest


TRT_LOGGER = trt.Logger()
yolo_checkpoint = "/home/jake/robotai/_checkpoints/yolov7-tiny.pt"

class TestTensorRT(unittest.TestCase):
    def test_basic(self):
        engine_path = yolo_checkpoint.replace(".pt", ".engine")

        with open(engine_path, "rb") as f:
                serialized_engine = f.read()

        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:

            engine = runtime.deserialize_cuda_engine(serialized_engine)
            context = engine.create_execution_context()

            for binding in engine:
                print(binding)

            input_idx = engine["images"]
            output_idx = engine["output"]
            input_dims = engine.get_binding_shape(input_idx)
            output_dims = engine.get_binding_shape(output_idx)

            print("input_dims:", input_dims)
            print("output_dims:", output_dims)

            bindings = []
            for binding in engine:
                shape = engine.get_binding_shape(binding)
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                data = torch.zeros(*shape, device="cuda:0")
                bindings.append(data)


            context.execute_v2([b.data_ptr() for b in bindings])

            print("bindings:", bindings)
