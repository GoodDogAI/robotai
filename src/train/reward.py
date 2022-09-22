import torch
import numpy as np
import torchvision

yolov7_class_num = 80

yolov7_class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


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

def _all_centers(bboxes: np.ndarray) -> np.ndarray:
    return np.sqrt(((bboxes[..., 0] - input_w / 2) / input_w) ** 2 +
                  ((bboxes[ ..., 1] - input_h / 2) / input_h) ** 2) + 0.1  # Small constant to prevent divide by zero explosion


def sum_centered_objects_present(bboxes: torch.FloatTensor) -> torch.float32:
    all_probs = bboxes[..., 4] * np.amax(bboxes[..., 5:], axis=-1)
    all_centers = _all_centers(bboxes)

    return np.sum(all_probs / all_centers)


def prioritize_centered_spoons_with_nms(bboxes: np.ndarray) -> float:
    bboxes = non_max_supression(bboxes)
    return prioritize_centered_objects(bboxes, class_weights={
        "person": 3,
        "spoon": 10,
    })


def prioritize_centered_objects(bboxes: np.ndarray, class_weights: dict) -> float:
    all_probs = bboxes[..., 4] * np.amax(bboxes[..., 5:], axis=-1)
    all_centers = _all_centers(bboxes)

    classes = np.argmax(bboxes[..., 5:], axis=-1)
    factors = np.ones_like(all_probs)

    for (cls, factor) in class_weights.items():
        factors *= np.where(classes == class_names.index(cls), factor, 1.0)

    return np.sum((all_probs * factors) / all_centers) * GLOBAL_REWARD_SCALE


def prioritize_centered_objects_with_nms(bboxes: np.ndarray) -> float:
    bboxes = non_max_supression(bboxes)
    return prioritize_centered_objects(bboxes, class_weights={
        "person": 3,
        "spoon": 10,
    })

class SumCenteredObjectsPresentReward(torch.nn.Module):
    reward_scale: 1.0

    def __init__(self, reward_scale):
        super().__init__()
        self.reward_scale = reward_scale

    def forward(self, bboxes: torch.FloatTensor) -> torch.Tensor:
        return torch.tensor(sum_centered_objects_present(bboxes)) * self.reward_scale