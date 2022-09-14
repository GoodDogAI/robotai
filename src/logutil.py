import os
import hashlib
import json
import logging
import numpy as np

from einops import rearrange
from typing import List, Dict, BinaryIO
from functools import total_ordering
from capnp.lib import capnp
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, parse_file_as
from cereal import log
from src.video import get_image_packets, decode_last_frame
from src.train.modelloader import load_vision_model
from contextlib import ExitStack

import src.PyNvCodec as nvc

logger = logging.getLogger(__name__)
DATA_FILE = "_hashes.json"


def sha256(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()



def validate_log(f: BinaryIO) -> bool:
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
                    # TODO Assert type and shape
                    logged_intermediate = np.frombuffer(evt.modelValidation.data, dtype=np.float32)
                    logged_intermediate = np.reshape(logged_intermediate, evt.modelValidation.shape)
                    y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
                    uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")
                    trt_outputs = valid_engine.infer({"y": y, "uv": uv})

                    diff = np.abs(trt_outputs["intermediate"] -n)
                    print(f"Model outputs: {trt_outputs}")
                    

                    # Compare the output to the expected output
                    

        return True
    except capnp.KjException:
        return False

@total_ordering
class LogSummary(BaseModel):
    filename: str
    sha256: str
    last_modified: int

    def __le__(self, other):
        return self.filename < other.filename


# Allows for quick and cached access to the SHA256 hash of a bunch of log files
class LogHashes:
    dir: str
    extension: str = ".log"
    files: Dict[str, LogSummary] = {}

    def __init__(self, dir):
        self.dir = dir
        self.update()

    def update(self):
        path = os.path.join(self.dir, DATA_FILE)
        if os.path.exists(path):
            self.existing_files = parse_file_as(Dict[str, LogSummary], path)
        else:
            self.existing_files = {}

        self.files = {}

        for file in os.listdir(self.dir):
            if not file.endswith(self.extension):
                continue
            filepath = os.path.join(self.dir, file)
            mtime = round(os.path.getmtime(filepath) * 1e9)

            if file in self.existing_files and self.existing_files[file].last_modified == mtime:
                self.files[file] = self.existing_files[file]
                continue

            logger.info(f"Hashing {filepath}")
            self.files[file] = LogSummary(filename=file, sha256=sha256(filepath), last_modified=mtime)

        with open(path + "_temp", "w") as f:
            json.dump(jsonable_encoder(self.files), f)

        if os.path.exists(path):
            os.remove(path)
        os.rename(path + "_temp", path)

    def values(self) -> List[LogSummary]:
        return list(self.files.values())

    def hash_exists(self, hash:str) -> bool:
        return hash in [f.sha256 for f in self.files.values()]

    def filename_exists(self, filename:str) -> bool:
        return filename in self.files

    def __str__(self):
        return str(self.files)
