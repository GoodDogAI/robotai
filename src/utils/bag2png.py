import os
import glob
import png
import rosbag
import numpy as np
from cereal import log
from src.video import create_video
from src.include.config import load_realtime_config
import PIL
from PIL import Image

camera_topic = ["/camera/rgb/image_rect_raw"]
CONFIG = load_realtime_config()

DECODE_WIDTH = int(CONFIG["CAMERA_WIDTH"])
DECODE_HEIGHT = int(CONFIG["CAMERA_HEIGHT"])

def extract(bag_file: str):
    print(f"Extracting {bag_file}")
    index = 0
    video_width, video_height = None, None

    with rosbag.Bag(bag_file, 'r') as bag:
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

                p = png.from_array(image_np, 'RGB', info={'bitdepth': 8})
                
                p.save(os.path.join(os.path.dirname(bag_file), f"_{index}.png"))
                index += 1

        


if __name__ == "__main__":
    extract("/media/storage/bagfiles/newbot_nov21/record-brain-sac-cosmic-cloud-483-26800-samp0.0_2021-12-29-16-21-04_0.bag")

