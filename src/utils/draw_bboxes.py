import numpy as np
from PIL import Image, ImageDraw,ImageFont

def draw_bboxes_pil(img: Image, bboxes, reward_config) -> Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    color = "red"

    # Internally, images are cropped down to the nearest multiple of the stride
    internal_width = img.width // reward_config["dimension_stride"] * reward_config["dimension_stride"]
    internal_height = img.height // reward_config["dimension_stride"] * reward_config["dimension_stride"]

    for row in bboxes:
        cxcywh = row[0:4]
        cxcywh[0] += (img.width - internal_width) / 2
        cxcywh[1] += (img.height - internal_height) / 2
        
        x1, y1 = cxcywh[0] - cxcywh[2] / 2, cxcywh[1] - cxcywh[3] / 2
        x2, y2 = cxcywh[0] + cxcywh[2] / 2, cxcywh[1] + cxcywh[3] / 2

        if row[4] > 0.15:
            label = reward_config["class_names"][np.argmax(row[5:])]
            txt_width, txt_height = font.getsize(label)
            draw.rectangle([x1, y1, x2, y2], width=1, outline=color)  # plot
            print("Found bbox:", cxcywh, label)

            draw.rectangle([x1, y1 - txt_height + 4, x1 + txt_width, y1], fill=color)
            draw.text((x1, y1 - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return img