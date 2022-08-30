import sys
# Hack to allow loading the pickled yolov7 model
sys.path.insert(0, "src/train/yolov7")

import time
import torch
import onnx
import numpy as np
import tensorrt as trt

from itertools import chain
from src.train.yolov7.models.experimental import attempt_load
from src.train.yolov7.utils.activations import Hardswish, SiLU
from src.train.yolov7.utils.general import set_logging, check_img_size

yolo_checkpoint = "/home/jake/robotai/_checkpoints/yolov7-tiny.pt"
batch_size = 1
img_size = (720, 1280)
device = "cuda:0"


# Load the original pytorch model
model = attempt_load(yolo_checkpoint)
labels = model.names

# Checks
gs = int(max(model.stride))  # grid size (max stride)
img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection
model.model[-1].export = False  # set Detect() layer grid export
y = model(img)  # dry run


# Convert that to ONNX
onnx_path = yolo_checkpoint.replace(".pt", ".onnx")
print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
model.eval()
output_names = ['classes', 'boxes'] if y is None else ['output']


torch.onnx.export(model, img, onnx_path, verbose=False, opset_version=12, input_names=['images'],
                  output_names=output_names,
                  dynamic_axes=None)

# Checks
onnx_model = onnx.load(onnx_path)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print("Confirmed ONNX model is valid")
    
# Convert the ONNX to TensorRT
engine_path = yolo_checkpoint.replace(".pt", ".engine")

print('\nStarting TensorRT export with onnx %s...' % onnx.__version__)
TRT_LOGGER = trt.Logger(trt.ILogger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
     builder.create_builder_config() as config, \
     trt.OnnxParser(network, TRT_LOGGER) as parser, \
     trt.Runtime(TRT_LOGGER) as runtime:
     

        # Don't use TF32 because it may give different results, and we are going for exact reproducability
        config.clear_flag(trt.BuilderFlag.TF32)

        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1
        
        with open(onnx_path, "rb") as onnx_file:
            print("Beginning ONNX file parsing")
            if not parser.parse(onnx_file.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
    
        
        print("Completed parsing of ONNX file")
        print("Beginning engine build")
        # plan = builder.build_serialized_network(network, config)
        # engine = runtime.deserialize_cuda_engine(plan)
        # print("Completed creating Engine")
        # with open(engine_path, "wb") as f:
        #     f.write(plan)
            
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        bindings = []
        for binding in engine:
            shape = engine.get_binding_shape(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data = torch.randn(*shape, device="cuda:0")
            bindings.append(data)


        start = time.perf_counter()
        for i in range(100):
            context.execute_v2([b.data_ptr() for b in bindings])
        print(f"TRT Took {(time.perf_counter() - start) / 100:0.4f} seconds per iteration")

        import onnxruntime
        ort_sess = onnxruntime.InferenceSession(onnx_path)
        ort_outputs = ort_sess.run(None, {'images': bindings[0].cpu().numpy()})

        start = time.perf_counter()
        for i in range(100):
            official = model(bindings[0])
        print(f"PT Took {(time.perf_counter() - start) / 100:0.4f} seconds per iteration")
    

        trt_close = torch.isclose(official[0], bindings[4], atol=1e-5, rtol=1e-3)
        onnx_close = torch.isclose(torch.from_numpy(ort_outputs[0]), bindings[4].cpu(), atol=1e-5, rtol=1e-3)

        print(f"TensorRT-PT Percent match: {trt_close.sum() / torch.numel(trt_close):.3f}")
        print(f"ONNX-TensorRT Percent match: {onnx_close.sum() / torch.numel(onnx_close):.3f}")

        onnx_close = torch.isclose(torch.from_numpy(ort_outputs[0]), official[0].cpu(), atol=1e-5, rtol=1e-3)

        print(f"ONNX-PT Percent match: {onnx_close.sum() / torch.numel(onnx_close):.3f}")

        # Check that the outputs are the same
        MODEL_MATCH_ATOL = 1e-5
        MODEL_MATCH_RTOL = 1e-3


        for i, torch_output in enumerate(chain(*official)):
            torch_output = torch_output
            ort_output = ort_outputs[i]
            #assert np.allclose(torch_output, ort_output, rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL), f"Output mismatch {i}"
            matches = torch.isclose(torch.from_numpy(ort_output), torch_output.cpu(), rtol=MODEL_MATCH_RTOL, atol=MODEL_MATCH_ATOL).sum()
            print(f"Output {i} matches: {matches / torch.numel(torch_output):.2%}")


        print("Done")

