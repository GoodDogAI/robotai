import numpy as np

from einops import rearrange
from typing import List, Dict, BinaryIO
from capnp.lib import capnp
from cereal import log
from src.config.config import DEVICE_CONFIG
from src.video import get_image_packets, decode_last_frame
from src.train.modelloader import load_vision_model, load_all_models_in_log
from contextlib import ExitStack
import src.PyNvCodec as nvc


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


ValidationStatus = "log.ModelValidation.ValidationStatus"

def aggregate_log_validation_status(stats: List[ValidationStatus]) -> str:
    if log.ModelValidation.ValidationStatus.validatedFailed in stats:
        return "validatedFailed"
    elif log.ModelValidation.ValidationStatus.validatedPassed in stats:
        return "validatedPassed"
    else:
        return "validatedSkipped"


def get_log_validation_status(log_file: str) -> str:
    try:
        with open(log_file, "rb") as f:
            events = log.Event.read_multiple(f)
            validation_stats: List[ValidationStatus] = []

            for evt in events:
                if evt.which() == "modelValidation":
                    validation_stats.append(evt.modelValidation.serverValidated)

            return aggregate_log_validation_status(validation_stats)
    except FileNotFoundError:
        return "validatedSkipped"


# Fully checks the log file, including validating any modelValidation type events
def full_validate_log(input: BinaryIO, output: BinaryIO) -> ValidationStatus:
    validation_stats: List[ValidationStatus] = []

    with load_all_models_in_log(input) as models:
        input.seek(0)

        try:
            events = log.Event.read_multiple(input)
            
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

                        # Run the model on the frame
                        logged_intermediate = np.array(list(evt.modelValidation.data), dtype=np.float32)
                        logged_intermediate = np.reshape(logged_intermediate, evt.modelValidation.shape)
                        y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
                        uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")
                        trt_outputs = models[evt.modelValidation.modelFullName].infer({"y": y, "uv": uv})
                        trt_intermediate = trt_outputs["intermediate"]

                        # Compare the output to the expected output
                        cos_sim = cosine_similarity(logged_intermediate.flatten(), trt_intermediate.flatten())
                        print(f"intermediate cosine similarity: {cos_sim}")
                        evt.modelValidation.serverSimilarity = float(cos_sim)

                        if cos_sim > .985:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedPassed
                        else:
                            evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedFailed
                    except KeyError:
                        print(f"Frame {evt.modelValidation.frameId} not found in log")
                        evt.modelValidation.serverValidated = log.ModelValidation.ValidationStatus.validatedSkipped

                elif evt.which() == "modelValidation" and \
                    evt.modelValidation.modelType == log.ModelValidation.ModelType.visionInput:
                    print(f"Checking vision input {evt.modelValidation.modelFullName} on frame {evt.modelValidation.frameId}...")

                    if evt.modelValidation.shape != [1, 1, 2, DEVICE_CONFIG.CAMERA_WIDTH]:
                        continue

                    # Render the video frame which is being referred to
                    try:
                        packets = get_image_packets(input.name, evt.modelValidation.frameId)
                        y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)

                        y_slice = y[:2, :]

                        logged_y_slice = np.array(list(evt.modelValidation.data), dtype=np.float32)
                        logged_y_slice = np.reshape(logged_y_slice, evt.modelValidation.shape)

                        cos_sim = cosine_similarity(logged_y_slice.flatten(), y_slice.flatten())
                        print(f"y value cosine similarity: {cos_sim}")

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

            return aggregate_log_validation_status(validation_stats)
        except capnp.KjException as ex:
            return log.ModelValidation.ValidationStatus.validatedFailed