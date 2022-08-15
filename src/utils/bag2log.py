import os
import rosbag
from cereal import log

camera_topic = ["/camera/rgb/image_rect_raw"]

def convert(bag_file: str, logfile: str):
    bag = rosbag.Bag(bag_file, 'r')

    for topic, msg, ts in bag.read_messages():
        print(msg)

if __name__ == "__main__":
    convert("/home/jake/bagfiles/newbot_nov21/record-brain-sac-atomic-cloud-515-20000-samp0.0_2022-05-08-16-29-52_0.bag", "/home/jake/robotairecords/converted/record-brain-sac-atomic-cloud-515-20000-samp0.0_2022-05-08-16-29-52_0.log")

