import os
import rosbag
import numpy as np
from cereal import log
from src.video import create_video
from src.include.config import load_realtime_config
from skimage.transform import rescale

camera_topic = ["/camera/rgb/image_rect_raw"]
CONFIG = load_realtime_config()

DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])

def convert(bag_file: str, logfile: str):
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

        all_frames = all_frames[:10]

        for frame in all_frames:
            frame = frame.reshape((video_height, video_width, -1))
            resized = rescale(frame, scale, channel_axis=2)
            if resized.shape[0] != DECODE_HEIGHT:
                l = (resized.shape[0] - DECODE_HEIGHT) // 2
                resized = resized[l:resized.shape[0] - l,]
            if resized.shape[1] != DECODE_WIDTH:
                l = (resized.shape[1] - DECODE_WIDTH) // 2
                resized = resized[:, l:resized.shape[1] - l]

            resized = resized.reshape((resized.shape[0], -1))
            resized = np.round(resized * 255).astype(np.uint8)
            resized_frames.append(resized)

        video_packets = create_video(resized_frames)

        for packet in video_packets:
            evt = log.Event.new_message()
            dat = evt.init("headEncodeData")
            dat.data = packet
            evt.write(lf)

if __name__ == "__main__":
    convert("/home/jake/bagfiles/newbot_nov21/record-brain-sac-atomic-cloud-515-20000-samp0.0_2022-05-08-16-29-52_0.bag", "/home/jake/robotairecords/record-brain-sac-atomic-cloud-515-20000-samp0.0_2022-05-08-16-29-52_0.log")

