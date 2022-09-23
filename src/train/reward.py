import torch
import numpy as np
import torchvision


def non_max_supression(input_bboxes: np.ndarray, iou_threshold: float = 0.50) -> np.ndarray:
    assert input_bboxes.shape[0] == 1, "This operation cannot be batched"

    yolo_bboxes = torch.from_numpy(input_bboxes)[0]

    boxes = torchvision.ops.box_convert(yolo_bboxes[..., 0:4], in_fmt="cxcywh", out_fmt="xyxy")
    scores = yolo_bboxes[..., 4] * torch.amax(yolo_bboxes[..., 5:], dim=-1)

    # Remove boxes that are below some low threshold
    confidence_filter_mask = scores > 0.10
    boxes = boxes[confidence_filter_mask]
    scores = scores[confidence_filter_mask]
    original_indexes = torch.arange(0, yolo_bboxes.shape[0], dtype=torch.int64)[confidence_filter_mask]

    nms_boxes = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
    original_nms_boxes = original_indexes[nms_boxes]

    return input_bboxes[0, original_nms_boxes]

def _all_centers(bboxes: torch.FloatTensor, width: int, height: int, center_epsilon: float) -> torch.FloatTensor:
    return torch.sqrt(((bboxes[..., 0] - width / 2) / width) ** 2 +
                  ((bboxes[ ..., 1] - height / 2) / height) ** 2) + center_epsilon  # Small constant to prevent divide by zero explosion



def prioritize_centered_objects(bboxes: np.ndarray, class_weights: dict) -> float:
    all_probs = bboxes[..., 4] * np.amax(bboxes[..., 5:], axis=-1)
    all_centers = _all_centers(bboxes)

    classes = np.argmax(bboxes[..., 5:], axis=-1)
    factors = np.ones_like(all_probs)

    for (cls, factor) in class_weights.items():
        factors *= np.where(classes == class_names.index(cls), factor, 1.0)

    return np.sum((all_probs * factors) / all_centers) * GLOBAL_REWARD_SCALE


class SumCenteredObjectsPresentReward(torch.nn.Module):
    reward_scale: 1.0

    def __init__(self, width, height, reward_scale, center_epsilon=0.1):
        super().__init__()
        self.width = width
        self.height = height
        self.reward_scale = reward_scale
        self.center_epsilon = center_epsilon

    def forward(self, bboxes: torch.FloatTensor) -> torch.Tensor:
        all_probs = bboxes[..., 4] * torch.amax(bboxes[..., 5:], dim=-1)
        all_centers = _all_centers(bboxes, self.width, self.height, self.center_epsilon)
        
        sum = torch.sum(all_probs / all_centers)

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