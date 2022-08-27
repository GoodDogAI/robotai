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

            # input_idx = engine[input_name]
            # output_idx = engine[output_name]

            # buffers = [None] * 2 # Assuming 1 input and 1 output
            # buffers[input_idx] = input_ptr
            # buffers[output_idx] = output_ptr

            # context.execute_v2(buffers, stream_ptr)
