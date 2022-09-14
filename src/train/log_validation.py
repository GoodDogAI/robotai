import numpy as np

from einops import rearrange
from typing import List, Dict, BinaryIO
from capnp.lib import capnp
from cereal import log
from src.video import get_image_packets, decode_last_frame
from src.train.modelloader import load_vision_model
from contextlib import ExitStack
import src.PyNvCodec as nvc

# Fully checks the log file, including validating any modelValidation type events
def full_validate_log(f: BinaryIO) -> bool:
    try:
        events = log.Event.read_multiple(f)
        
        with ExitStack() as stack:
            valid_engine = stack.enter_context(load_vision_model("yolov7-tiny-s53"))

            for evt in events:
                evt.which()

                # Now, also process modelValidation events, and check if they are valid
                if evt.which() == "modelValidation" and \
                    evt.modelValidation.modelType == log.ModelValidation.ModelType.visionIntermediate:
                    print(f"Checking vision model {evt.modelValidation.modelFullName} on frame {evt.modelValidation.frameId}...")

                    # Render the video frame which is being referred to
                    packets = get_image_packets(f.name, evt.modelValidation.frameId)
                    y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)

                    # TODO: If you didn't find a packet, then it's an error, unless this is the last modelValidation message, and 
                    # then, due to encoding delays, you may expect that frame to come in on a later log rotation, so it's okay to skip it

                    # Load in the model runner for the model in question


                    # Run the model on the frame
                    logged_intermediate = np.array(list(evt.modelValidation.data), dtype=np.float32)
                    logged_intermediate = np.reshape(logged_intermediate, evt.modelValidation.shape)
                    y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
                    uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")
                    trt_outputs = valid_engine.infer({"y": y, "uv": uv})
                    trt_intermediate = trt_outputs["intermediate"]

                    # Compare the output to the expected output
                    diff = np.abs(trt_outputs["intermediate"] - logged_intermediate)
                    matches = np.isclose(trt_intermediate, logged_intermediate, rtol=1e-2, atol=1e-2).sum()
                    print(f"Logged Output matches: {matches / logged_intermediate.size:.3%}")

                elif evt.which() == "modelValidation" and \
                    evt.modelValidation.modelType == log.ModelValidation.ModelType.visionInput:
                    print(f"Checking vision input {evt.modelValidation.modelFullName} on frame {evt.modelValidation.frameId}...")

                    # Render the video frame which is being referred to
                    packets = get_image_packets(f.name, evt.modelValidation.frameId)
                    y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)

                    logged_y_slice = np.array(list(evt.modelValidation.data), dtype=np.float32)
                    logged_y_slice = np.reshape(logged_y_slice, evt.modelValidation.shape)
                    print("A")


                    
                    

        return True
    except capnp.KjException:
        return False