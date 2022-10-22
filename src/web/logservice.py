import io
import os
import tempfile
import av
import hashlib
import shutil
import numpy as np

from typing import List

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

from cereal import log
from src.config import HOST_CONFIG
from src.config.config import MODEL_CONFIGS
from src.train.modelloader import load_vision_model
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
    return lh.group_logs()

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

        for evt in events:
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
            result.append(data)

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

    try:
        packets = get_image_packets(os.path.join(lh.dir, logfile), frameid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found")
        
    y, uv = decode_last_frame(packets, pixel_format=nvc.PixelFormat.NV12)
    y = rearrange(y.astype(np.float32), "h w -> 1 1 h w")
    uv = rearrange(uv.astype(np.float32), "h w -> 1 1 h w")
    
    # Load the default reward model
    from src.train.modelloader import model_fullname, update_model_config_caches
    update_model_config_caches()
    reward_config = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_REWARD_CONFIG]
    fullname = model_fullname(reward_config)

    with load_vision_model(fullname) as model:
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


@router.get("/{logfile}/thumbnail")
def get_log_thumbnail(logfile: str, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    with open(os.path.join(lh.dir, logfile), "rb") as f:
        events = log.Event.read_multiple(f)

        for i, evt in enumerate(events):
            if evt.which() == "headEncodeData":
                if evt.headEncodeData.idx.flags & 0x8:
                    packet = evt.headEncodeData.data
                    return Response(content=packet, media_type="image/heic")

    raise HTTPException(status_code=500, detail="Unable to find thumbnail")

@router.get("/{logfile}/video")
def get_log_video(logfile: str, lh: LogHashes = Depends(get_loghashes)):
    if not lh.filename_exists(logfile):
        raise HTTPException(status_code=404, detail="Log not found")

    with tempfile.NamedTemporaryFile(suffix=".mkv") as vf, \
         open(os.path.join(lh.dir, logfile), "rb") as f:

        output = av.open(vf.name, "w")
        out_stream = output.add_stream("hevc", 15)
        events = log.Event.read_multiple(f)
        packet_index = 0

        for i, evt in enumerate(events):
            if evt.which() == "headEncodeData":
                packet = av.Packet(evt.headEncodeData.data)
                packet.stream = out_stream
                packet.pts = packet_index / 15.0
                packet.dts = packet_index / 15.0
                packet_index += 1
                output.mux(packet)

        output.close()

        vf.seek(0)
        return Response(content=vf.read(), media_type="video/x-matroska", headers={"Content-Disposition": f"attachment; filename={logfile}.mkv"})  