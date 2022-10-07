import os
from src.config.dotdict import dotdict

DEVICE_CONFIG = dotdict({
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 15,

    "LOG_DURATION_SECONDS": 60,
    "LOG_PATH": "/media/card",
    "LOG_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    "MODEL_STORAGE_PATH": "/home/robot/models",
    "MODEL_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    # 0 QP is nearly lossless, 50 is a huge lossy compression, we don't specify a bitrate because
    # we want each frame to be encoded to some minimum quality level
    "ENCODER_QP": 10, 
    "ENCODER_DEV": "/dev/nvhost-msenc",
    "ENCODER_COMP_NAME": "NVENC",

    "AUDIO_DEVICE_NAME": "hw:APE,0",
    "AUDIO_PCM_FORMAT": "S32_LE",
    "AUDIO_SAMPLE_RATE": 48_000,
    "AUDIO_CHANNELS": 2,

    "ODRIVE_SERIAL_PORT": "/dev/ttyACM0",
    "ODRIVE_BAUD_RATE": 115_200,
    "ODRIVE_MAX_SPEED": 2.0,
 
    "SIMPLEBGC_SERIAL_PORT": "/dev/ttyTHS0",
    "SIMPLEBGC_BAUD_RATE": 115_200,
})


HOST_CONFIG = dotdict({
    "RECORD_DIR": os.environ.get("ROBOTAI_RECORD_DIR", "/media/storage/robotairecords"),
    "CACHE_DIR": os.environ.get("ROBOTAI_CACHE_DIR", "/media/storage/robotaicache"),

    "DEFAULT_BRAIN_CONFIG": "orange-hippo-1",
    "DEFAULT_REWARD_CONFIG": "yolov7-tiny-prioritize_centered_nms",

    "DEFAULT_DECODE_GPU_ID": 0,
})

BRAIN_CONFIGS = dotdict({
    "orange-hippo-1": {
        "models": {
            #"brain_model": "orange-hippo-1",
            "vision_model": "yolov7-tiny-s53",
        },

        "audio_resampler": "test1",
        "msgvec": {
            "obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -1,
                    "timeout": 0.20,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.20,
                    "filter": {
                        "field": "voltage.type",
                        "op": "eq",
                        "value": "mainBattery",
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 13.5],
                        "vec_range": [-1, 1],
                    }
                },
                
                { 
                    "type": "msg",
                    "path": "headFeedback.pitchAngle",
                    "index": -1,
                    "timeout": 0.20,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 17003,
                    "index": -1,
                }
            ],

            "act": [
                {
                    "type": "msg",
                    "path": "odriveCommand.leftMotor.vel",
                    "timeout": 0.20,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "index": -1,
                    "timeout": 0.20,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ],
        }
    }
})

YOLOV7_CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
  "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# Configures how models turn from pytorch checkpoints into runnable objects on the host and device
MODEL_CONFIGS = dotdict({
    "yolov7-tiny-s53": {
        "type": "vision",
        "load_fn": "src.models.yolov7.load.load_yolov7",
        "input_format": "rgb",
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": "input.219", # Another option to try could be onnx::Conv_254
        "intermediate_slice": 53,
    },

    "yolov7-tiny-final3-s53": {
        "type": "vision",
        "load_fn": "src.models.yolov7.load.load_yolov7",
        "input_format": "rgb",
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-tiny.pt",

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "intermediate_layer": ["onnx::Conv_351", "onnx::Conv_379", "onnx::Conv_365"], 
        "intermediate_slice": 53,
    },

    "yolov7-tiny-prioritize_centered_nms": {
        "type": "reward",
        "load_fn": "src.models.yolov7.load.load_yolov7",
        "input_format": "rgb",
        "checkpoint": "/home/jake/robotai/_checkpoints/yolov7-w6.pt",
        "class_names": YOLOV7_CLASS_NAMES,

        # Input dimensions must be divisible by the stride
        # In current situations, the image will be cropped to the nearest multiple of the stride
        "dimension_stride": 32,

        "max_detections": 100,
        "iou_threshold": 0.45,

        "reward_module": "src.train.reward.SumCenteredObjectsPresentReward",

        "reward_kwargs": {
            "class_weights": {
                "person": 10,
                "cat": 5,
                "dog": 5,
                "bird": 5,
                "horse": 5,
                "sheep": 5,
                "cow": 5,
                "elephant": 5,
                "bear": 5,
                "zebra": 5,
                "giraffe": 5,
                "tv": 0.5,
                "book": 0.1,
            },
            "reward_scale": 0.10,
            "center_epsilon": 0.1,  
        }
    }
})


WEB_CONFIG = dotdict({
   # Unknown
})

