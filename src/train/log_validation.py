import numpy as np

from einops import rearrange
from typing import List, Dict, BinaryIO
from capnp.lib import capnp
from cereal import log
from src.video import get_image_packets, decode_last_frame
from src.train.modelloader import load_vision_model
from contextlib import ExitStack
import src.PyNvCodec as nvc


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


ValidationStatus = "log.ModelValidation.ValidationStatus"

def get_log_validation_status(stats: List[ValidationStatus]) -> ValidationStatus:
    if log.ModelValidation.ValidationStatus.validatedFailed in stats:
        return log.ModelValidation.ValidationStatus.validatedFailed
    elif log.ModelValidation.ValidationStatus.validatedPassed in stats:
        return log.ModelValidation.ValidationStatus.validatedPassed
    else:
        return log.ModelValidation.ValidationStatus.validatedSkipped

# Fully checks the log file, including validating any modelValidation type events
def full_validate_log(input: BinaryIO, output: BinaryIO) -> ValidationStatus:
    validation_stats: List[ValidationStatus] = []

    try:
        events = log.Event.read_multiple(input)
        
        with ExitStack() as stack:
            valid_engine = stack.enter_context(load_vision_model("yolov7-tiny-s53"))

            for evt in events:
                evt.which()
                evt = evt.as_builder()

                # Now, also process modelValidation events, and check if they are valid
                if evt.which() == "modelValidation" and \
                    evt.modelValidation.modelType == log.ModelValidation.ModelType.visionIntermediate:
                    print(f"Checking vision model {evt.modelValidation.modelFullName} on frame {evt.modelValidation.frameId}...")

                    # Render the video frame which is being referred to
                    try:
                        packets = get_image_packets(input.name, evt.modelValidation.frameId)

                        y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)

                        # TODO Better control over which model is loaded, and then only load it once?
                        # Load in the model runner for the model in question

                        # Run the model on the frame
                        logged_intermediate = np.array(list(evt.modelValidation.data), dtype=np.float32)
                        logged_intermediate = np.reshape(logged_intermediate, evt.modelValidation.shape)
                        y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
                        uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")
                        trt_outputs = valid_engine.infer({"y": y, "uv": uv})
                        trt_intermediate = trt_outputs["intermediate"]

                        # Compare the output to the expected output
                        cos_sim = cosine_similarity(logged_intermediate.flatten(), trt_intermediate.flatten())
                        print(f"intermediate cosine similarity: {cos_sim}")
                        evt.modelValidation.serverSimilarity = float(cos_sim)

                        if cos_sim > .99:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedPassed
                        else:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedFailed
                    except KeyError:
                        print(f"Frame {evt.modelValidation.frameId} not found in log")
                        evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedSkipped

                elif evt.which() == "modelValidation" and \
                    evt.modelValidation.modelType == log.ModelValidation.ModelType.visionInput:
                    print(f"Checking vision input {evt.modelValidation.modelFullName} on frame {evt.modelValidation.frameId}...")

                    # Render the video frame which is being referred to
                    try:
                        packets = get_image_packets(input.name, evt.modelValidation.frameId)
                        y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)

                        y_slice = y[:2, :]

                        logged_y_slice = np.array(list(evt.modelValidation.data), dtype=np.float32)
                        logged_y_slice = np.reshape(logged_y_slice, evt.modelValidation.shape)

                        cos_sim = cosine_similarity(logged_y_slice.flatten(), y_slice.flatten())
                        print(f"y value cosine similarity: {cos_sim}")

                        evt.modelValidation.serverValidated = True
                        evt.modelValidation.serverSimilarity = float(cos_sim)

                        if cos_sim > .99:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedPassed
                        else:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedFailed

                    except KeyError:
                        evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedSkipped
                        print(f"Frame {evt.modelValidation.frameId} not found in log")

                if evt.which() == "modelValidation":
                    validation_stats.append(evt.modelValidation.serverValidated)

                # Write the log entry to the output file
                evt.write(output)

        return get_log_validation_status(validation_stats)
    except capnp.KjException as ex:
        return log.ModelValidation.ValidationStatus.validatedFailed