import sys
# Hack to allow loading the pickled yolov7 model
sys.path.insert(0, "src/train/yolov7")

import torch
import onnx
import tensorrt as trt

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
TRT_LOGGER = trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
     builder.create_builder_config() as config, \
     trt.OnnxParser(network, TRT_LOGGER) as parser, \
     trt.Runtime(TRT_LOGGER) as runtime:
     
        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1
        
        with open(onnx_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
    
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        #network.get_input(0).shape = [1, 3, 608, 608]
        print("Completed parsing of ONNX file")
        print("Beginning engine build")
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_path, "wb") as f:
            f.write(plan)
            

# Run both on the same random input and make sure
