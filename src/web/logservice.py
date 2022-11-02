import functools
import io
import os
import subprocess
import time
import tempfile
import hashlib
import shutil
import numpy as np

from typing import List, Literal

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse, Response

from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

from cereal import log
from src.config import HOST_CONFIG, MODEL_CONFIGS, DEVICE_CONFIG
from src.train.modelloader import cached_vision_model, model_fullname
from src.train.rldataset import MsgVecDataset
from src.utils.draw_bboxes import draw_bboxes_pil

from src.web.dependencies import get_loghashes
from src.logutil import LogHashes, LogSummary, quick_validate_log

from src.video import get_image_packets, decode_last_frame
from src.train import log_validation
import src.PyNvCodec as nvc

router = APIRouter(prefix="/logs",
    tags=["logs"],
)


async def asha256(fp: UploadFile) -> str:
    sha256_hash = hashlib.sha256()
    while byte_block := await fp.read(65536):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@router.get("/")
async def list_logs(lh: LogHashes = Depends(get_loghashes)) -> List[List[LogSummary]]:
    all_logs = lh.group_logs()

    # Update the validation metadata
    for log_group in all_logs:
        for log in log_group:
            if "validation" not in log.meta:
                lh.update_metadata(log.filename, validation=log_validation.get_log_validation_status(os.path.join(lh.dir, log.filename)))

    return all_logs


@router.get("/exists/{sha256}")
async def log_exists(sha256: str, lh: LogHashes = Depends(get_loghashes)):
    return lh.hash_exists(sha256)


@router.post("/")
async def post_log(logfile: UploadFile, sha256: str=Form(), lh: LogHashes = Depends(get_loghashes)):
    # Make sure the hash doesn't exist already
    local_hash = await asha256(logfile)

    if lh.hash_exists(local_hash):
        raise HTTPException(status_code=500, detail="Log hash already exists")

    if lh.filename_exists(logfile.filename):
        raise HTTPException(status_code=500, detail="Log filename already exists")

    if sha256 != local_hash:
        raise HTTPException(status_code=400, detail="SHA256 did not match expected value")

    logfile.file.seek(0, io.SEEK_END)
    if logfile.file.tell() < 1:
        raise HTTPException(status_code=400, detail="Empty file")

    # Check that you can read all messages
    logfile.file.seek(0)
    if not quick_validate_log(logfile.file):
        raise HTTPException(status_code=400, detail="Log file is not a serialized capnp buffer")

    # Determine a new filename
    newfilename = os.path.join(lh.dir, logfile.filename)
    if os.path.exists(newfilename):
        raise HTTPException(status_code=500, detail="Log filename already exists")

    # Copy to a tempfile, because capnp can't read full messages from a stream
    with tempfile.NamedTemporaryFile("w+b") as tf, open(newfilename, "wb") as f:
        logfile.file.seek(0)
        shutil.copyfileobj(logfile.file, tf)
        tf.seek(0)

        # Copy over to the final location doing a new validation
        log_validation.full_validate_log(tf, f)

    lh.update(original_hashes={
        logfile.filename: local_hash
    })
    await logfile.close()


@router.get("/{logfile}")
async def get_log(logfile: str, lh: LogHashes = Depends(get_loghashes)) -> JSONResponse:
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    result = []

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)
        headIndex = -1

        for index, evt in enumerate(events):
            which = evt.which()

            if which == "headEncodeData":
                headIndex = evt.headEncodeData.idx.frameId

            # Add in some sizing metadata
            result.append({
                "index": index,
                "which": which,
                "headIndex": headIndex,
                "total_size_bytes": evt.total_size.word_count * 8
            })

    # Don't try to encode raw data fields in the json,
    # it will just get interpreted as utf-8 text and you will have a bad time
    return JSONResponse(jsonable_encoder(result,
                        custom_encoder={bytes: lambda data_obj: None}))


@router.get("/{logfile}/entry/{index}")
async def get_log_entry(logfile: str, index: int, lh: LogHashes = Depends(get_loghashes)) -> JSONResponse:
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    result = None

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for i, evt in enumerate(events):
            if i == index:
                data = evt.to_dict()

                # Cut out some hard datafields in certain message types that would
                # otherwise make for huge downloads
                which = evt.which()
                if which == "micData":
                    del data["micData"]["data"]
                elif which == "modelValidation":
                    del data["modelValidation"]["data"]

                # Add in some sizing metadata
                data["_total_size_bytes"] = evt.total_size.word_count * 8
                result = data

    if result is None:
        raise HTTPException(status_code=404, detail="Log entry not found")

    # Don't try to encode raw data fields in the json,
    # it will just get interpreted as utf-8 text and you will have a bad time
    return JSONResponse(jsonable_encoder(result,
                                         custom_encoder={bytes: lambda data_obj: None}))


@router.get("/{logfile}/frame/{frameid}")
def get_log_frame(logfile: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    try:
        packets = get_image_packets(os.path.join(lh.dir, logfile), frameid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    rgb = decode_last_frame(packets, pixel_format=nvc.PixelFormat.RGB)
    img = Image.fromarray(rgb)
    img_data = io.BytesIO()
    img.save(img_data, format="JPEG")
    response = Response(content=img_data.getvalue(), media_type="image/jpeg")

    return response


@router.get("/{logfile}/depth/{frameid}")
def get_log_frame(logfile: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    try:
        packets = get_image_packets(os.path.join(lh.dir, logfile), frameid, type="depthEncodeData")
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    rgb = decode_last_frame(packets, pixel_format=nvc.PixelFormat.RGB)
    img = Image.fromarray(rgb)
    img_data = io.BytesIO()
    img.save(img_data, format="JPEG")
    response = Response(content=img_data.getvalue(), media_type="image/jpeg")

    return response


@router.get("/{logfile}/frame_reward/{frameid}")
def get_reward_frame(logfile: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")
    start = time.perf_counter()
    try:
        packets = get_image_packets(os.path.join(lh.dir, logfile), frameid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)
    y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
    uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")

    # Load the default reward model
    reward_config = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_REWARD_CONFIG]
    fullname = model_fullname(reward_config)

    with cached_vision_model(fullname) as model:
        trt_outputs = model.infer({"y": y, "uv": uv})
  
    # Draw the bounding boxes on a transparent PNG the same size as the main image
    img = Image.new("RGBA", (y.shape[3], y.shape[2]), (0, 0, 0, 0))
    draw_bboxes_pil(img, trt_outputs["bboxes"], reward_config)

    # Add in the reward for fun debugging
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    reward_label = f"Reward: {trt_outputs['reward']:.2f}"
    txt_width, txt_height = font.getsize(reward_label)
    draw.text((img.width - txt_width - 10, img.height - txt_height - 10), reward_label, fill=(255, 255, 255), font=font)

    img_data = io.BytesIO()
    img.save(img_data, format="PNG")
    response = Response(content=img_data.getvalue(), media_type="image/png")
    return response


@router.get("/{logfile}/video")
def get_log_video(logfile: str, camera: Literal["color", "depth"]="color", lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    assert camera in ["color", "depth"], "Invalid camera setting"

    with tempfile.TemporaryDirectory() as td, open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        # Read raw frame data into a file, creating a so called "annexb" file
        with open(os.path.join(td, "frames.hevc"), "wb") as vf:
            for i, evt in enumerate(events):
                if evt.which() == "headEncodeData" and camera == "color":
                    vf.write(evt.headEncodeData.data)
                elif evt.which() == "depthEncodeData" and camera == "depth":
                    vf.write(evt.depthEncodeData.data)

        # Convert that to a containerized video file by remuxing with ffmpeg
        # This is a bit of a hack, but it works
        subprocess.run(["ffmpeg", "-y", "-r", str(DEVICE_CONFIG.CAMERA_FPS), "-i", os.path.join(td, "frames.hevc"), "-map", "0", "-c", "copy", os.path.join(td, f"{logfile}.mp4")], check=True, stdout=subprocess.PIPE)

        with open(os.path.join(td, f"{logfile}.mp4"), "rb") as vf:
            return Response(content=vf.read(), media_type="video/mp4", headers={"Content-Disposition": f"attachment; filename={logfile}.mkv"})  

@functools.lru_cache(maxsize=8)
def _get_loggroup(logfile: str, model_name: str, lh: LogHashes):
    model_config = MODEL_CONFIGS[model_name]
    dataset = MsgVecDataset(lh.dir, model_config)

    # Get the loggroup which contains the current log file
    loggroup = None
    for lg in lh.group_logs():
        if logfile in [lf.filename for lf in lg]:
            loggroup = lg
            break

    if loggroup is None:
        raise HTTPException(status_code=404, detail="Log not found in group")
    
    return list(dataset.generate_log_group(loggroup, shuffle_within_group=False))

@router.get("/{logfile}/msgvec/{model_name}/{frameid}")
def get_msgvec(logfile: str, model_name: str, frameid: int, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model not found")

    target_key = f"{lh.get_logsummary(logfile).get_runname()}-{frameid}"

    print(f"Start key: {_get_loggroup(logfile, model_name, lh)[0]['key']}")
    print(f"End key: {_get_loggroup(logfile, model_name, lh)[-1]['key']}")

    # Run msgvec up to the desired frame
    for packet in _get_loggroup(logfile, model_name, lh):
        #print(packet["key"])
        if packet["key"] == target_key:
            return JSONResponse({
                "key": packet["key"],
                "obs": packet["obs"].tolist(),
                "act": packet["act"].tolist(),
                "reward": packet["reward"],
                "done": packet["done"],
            })

    raise HTTPException(status_code=404, detail="Frame not found")