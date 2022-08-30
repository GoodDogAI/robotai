import torch

def load_yolov7(checkpoint: str, device=None) -> torch.nn.Module:
    import sys
    # Hack to allow loading the pickled yolov7 model
    sys.path.insert(0, "src/train/yolov7")

    from src.train.yolov7.models.experimental import attempt_load
    model = attempt_load(checkpoint, map_location=device)

    model.model[-1].export = False
    model.eval()

    return model