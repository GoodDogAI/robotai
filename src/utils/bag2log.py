import os
import glob
import rosbag
import numpy as np
from cereal import log
from src.video import create_video, V4L2_BUF_FLAG_KEYFRAME
from src.config import DEVICE_CONFIG
import PIL
from PIL import Image

camera_topic = ["/camera/rgb/image_rect_raw"]

DECODE_WIDTH = DEVICE_CONFIG.CAMERA_WIDTH
DECODE_HEIGHT = DEVICE_CONFIG.CAMERA_HEIGHT

def convert(bag_file: str, logfile: str):
    print(f"Converting {bag_file} to {logfile}")
    all_frames = []
    video_width, video_height = None, None

    with rosbag.Bag(bag_file, 'r') as bag, open(logfile, "wb") as lf:
        for topic, msg, ts in bag.read_messages():
            if topic == "/processed_img":
                img = []
                for i in range(0, len(msg.data), msg.step):
                    img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))
                image_np = np.array(img)

                if video_width is None or video_height is None:
                    video_width, video_height = msg.width, msg.height
                else:
                    assert msg.width == video_width, "Video dims must be consistent"
                    assert msg.height == video_height, "Video dims must be consistent"

                all_frames.append(image_np)

        # Rescale and center crop the images so that they match the usual decoding width/height
        resized_frames = []
        scale = max(DECODE_WIDTH / video_width, DECODE_HEIGHT / video_height)

        for frame in all_frames:
            frame = frame.reshape((video_height, video_width, -1))
            frame = Image.fromarray(frame)
            resized = frame.resize((round(video_width * scale), round(video_height*  scale)), 
                                    resample=PIL.Image.Resampling.BILINEAR)
            if resized.height != DECODE_HEIGHT:
                l = (resized.height - DECODE_HEIGHT) // 2
                resized = resized.crop((0, l, resized.width, resized.height - l))
            if resized.width != DECODE_WIDTH:
                l = (resized.width - DECODE_WIDTH) // 2
                resized = resized.crop((l, 0, resized.width - l, resized.height))

            resized = np.array(resized)
            resized_frames.append(resized)

        video_packets = create_video(resized_frames)

        for idx, packet in enumerate(video_packets):
            evt = log.Event.new_message()
            dat = evt.init("headEncodeData")
            dat.data = packet

            if idx == 0:
                dat.idx.flags = V4L2_BUF_FLAG_KEYFRAME
            else:
                dat.idx.flags = 0

            dat.idx.frameId = idx
                
            evt.write(lf)

def _do_convert(path: str):
    convert(path, path.replace("/media/storage/bagfiles/newbot_nov21/", "/media/storage/robotairecords/converted/").replace(".bag", ".log"))

if __name__ == "__main__":
    from multiprocessing import Pool

    paths = glob.glob("/media/storage/bagfiles/newbot_nov21/*.bag")

    with Pool(processes=8) as pool:
        pool.map(_do_convert, paths)
       

