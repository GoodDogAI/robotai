import torch
import numpy as np
import torchvision
from typing import Dict, List


def _all_centers(bboxes: torch.FloatTensor, width: int, height: int, center_epsilon: float) -> torch.FloatTensor:
    return torch.sqrt(((bboxes[..., 0] - width / 2) / width) ** 2 +
                  ((bboxes[ ..., 1] - height / 2) / height) ** 2) + center_epsilon  # Small constant to prevent divide by zero explosion


class SumCenteredObjectsPresentReward(torch.nn.Module):
    def __init__(self, width, height, reward_scale=1.0, class_names: List[str]=None, class_weights: Dict[str, float]=None, center_epsilon=0.1):
        super().__init__()
        self.width = width
        self.height = height
        self.reward_scale = reward_scale
        self.class_names = class_names
        self.class_weights = class_weights
        self.center_epsilon = center_epsilon

        if self.class_names and self.class_weights:
            self.class_weight_data = []
            for cls in class_names:
                if cls in class_weights:
                    self.class_weight_data.append(class_weights[cls])
                else:
                    self.class_weight_data.append(1.0)

    def forward(self, bboxes: torch.FloatTensor) -> torch.Tensor:
        all_probs = bboxes[..., 4] * torch.amax(bboxes[..., 5:], dim=-1)
        all_centers = _all_centers(bboxes, self.width, self.height, self.center_epsilon)

        if self.class_weights is not None:
            _weight_data = torch.tensor(self.class_weight_data, dtype=torch.float32, device=bboxes.device)
            all_classes = torch.argmax(bboxes[..., 5:], dim=-1)
            all_class_weights = torch.gather(_weight_data, 0, all_classes)

            sum = torch.sum((all_probs * all_class_weights) / all_centers)
        else:
            sum = torch.sum((all_probs) / all_centers)

        return sum * self.reward_scale


class ConvertCropVisionReward(torch.nn.Module):
    def __init__(self, converter, cropper, vision_model, detection, reward_fn):
        super().__init__()
        self.converter = converter
        self.cropper = cropper
        self.vision_model = vision_model
        self.detection = detection
        self.reward_fn = reward_fn

    def forward(self, y=None, uv=None):
        img = self.converter(y, uv) 
        img = self.cropper(img)
        raw_detections, extra_layers = self.vision_model(img)
        bboxes = self.detection(raw_detections)
        reward = self.reward_fn(bboxes)
        return reward, bboxes, raw_detections


class ThresholdNMS(torch.nn.Module):
    def __init__(self, iou_threshold, max_detections):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, yolo_bboxes):
        boxes = yolo_bboxes[0, ..., 0:4]

        # Convert the boxes from cx, cy, w, h to xyxy
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                        dtype=torch.float32,
                                        device=boxes.device)
        boxes @= convert_matrix

        scores = yolo_bboxes[0, ..., 4] * torch.amax(yolo_bboxes[0, ..., 5:], dim=-1)
        nms_indexes = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_threshold)[:self.max_detections]
        return yolo_bboxes[0, nms_indexes]