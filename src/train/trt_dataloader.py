import numpy as np
import cv2




def load_data():
    img = cv2.imread("/home/jake/Downloads/horses.jpg") 
    img = cv2.resize(img, (1280, 736), interpolation=cv2.INTER_LINEAR)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB,
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0

    return [{
        "images": img
        }]


