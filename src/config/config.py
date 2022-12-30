import os
from src.config.dotdict import dotdict

DEVICE_CONFIG = dotdict({
    "CAMERA_WIDTH": 1280,
    "CAMERA_HEIGHT": 720,
    "CAMERA_FPS": 15,

    "CAMERA_GYRO_FPS": 200,
    "CAMERA_ACCEL_FPS": 200,

    "LOG_DURATION_SECONDS": 60,
    "LOG_PATH": "/media/card",
    "LOG_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    "MODEL_STORAGE_PATH": "/media/card/models",
    "MODEL_SERVICE": "http://jake-training-box.jakepoz.gmail.com.beta.tailscale.net:8000",

    # 0 QP is nearly lossless, 50 is a huge lossy compression, we don't specify a bitrate because
    # we want each frame to be encoded to some minimum quality level
    "ENCODER_HEAD_COLOR_QP": 10, 
    "ENCODER_HEAD_DEPTH_MAXBITRATE": 2_000_000,
    "ENCODER_DEV": "/dev/nvhost-msenc",
    "ENCODER_COMP_NAME": "NVENC",

    "OVERRIDE_LINEAR_SPEED": 0.50,
    "OVERRIDE_ANGULAR_SPEED": 0.15,

    "AUDIO_DEVICE_NAME": "hw:APE,0",
    "AUDIO_PCM_FORMAT": "S32_LE",
    "AUDIO_SAMPLE_RATE": 48_000,
    "AUDIO_CHANNELS": 2,

    "ODRIVE_SERIAL_PORT": "/dev/ttyACM0",
    "ODRIVE_BAUD_RATE": 115_200,
    "ODRIVE_MAX_SPEED": 2.0,
 
    "SIMPLEBGC_SERIAL_PORT": "/dev/ttyTHS0",
    "SIMPLEBGC_BAUD_RATE": 115_200,

    "SENSOR_REALSENSE_D455": 1,
    "SENSOR_REALSENSE_D435I": 2,
    "SENSOR_SIMPLEBGC_ICM2060X": 3,

    "SENSOR_TYPE_ACCELEROMETER": 1,
    "SENSOR_TYPE_GYRO": 2,
})


HOST_CONFIG = dotdict({
    "RECORD_DIR": os.environ.get("ROBOTAI_RECORD_DIR", "/media/storage/robotairecords"),
    "CACHE_DIR": os.environ.get("ROBOTAI_CACHE_DIR", "/media/storage/robotaicache"),

    "DEFAULT_BRAIN_CONFIG": "basic-brain-discrete-1",
    "DEFAULT_REWARD_CONFIG": "yolov7-tiny-prioritize_centered_nms",

    "DEFAULT_DECODE_GPU_ID": 0,
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

        "intermediate_layer": "/vision_model/model.59/Concat_output_0", # Another option to try could be onnx::Conv_254
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

        "intermediate_layer": ["/vision_model/model.74/act/LeakyRelu_output_0", "/vision_model/model.75/act/LeakyRelu_output_0", "/vision_model/model.76/act/LeakyRelu_output_0"], 
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
                "sports ball": 2.5,
                "tv": 0.5,
                "refrigerator": 0.5,
                "book": 0.1,
            },
            "reward_scale": 0.01,
            "center_epsilon": 0.1,  
        }
    },

    "basic-brain-test1": {
        "type": "brain",

        "checkpoint": "/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-run100.zip",
        "load_fn": "src.models.stable_baselines3.load.load_stable_baselines3_sac_actor",

        "models": {
            "vision": "yolov7-tiny-s53",
            "reward": "yolov7-tiny-prioritize_centered_nms",
        },

        "msgvec": {
            "obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.current",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.rightMotor.vel",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.rightMotor.current",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
                    "filter": {
                        "field": "voltage.type",
                        "op": "eq",
                        "value": "mainBattery",
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 15],
                        "vec_range": [-1, 1],
                    }
                },
                
                { 
                    "type": "msg",
                    "path": "headFeedback.pitchAngle",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.pitchMotorPower",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 1],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.yawAngle",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.yawMotorPower",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 1],
                        "vec_range": [-1, 1],
                    },
                },

                # These messages provide feedback on the last few commands send to the motors, as there is some delay between a desired action and its effect
                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityLeft",
                    "index": -5,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "index": -5,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "index": -5,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "index": -5,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.0",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.1",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.2",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.0",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.1",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.2",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 17003,
                    "timeout": 0.100,
                    "index": [-1, -2],
                }
            ],

            "act": [
                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityLeft",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-0.5, 0.5],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },
            ],

            "rew": {
                "base": "reward",

                "override": {
                    "positive_reward": 10.0,
                    "positive_reward_timeout": 2.0,

                    "negative_reward": -15.0,
                    "negative_reward_timeout": 2.0,
                }
            },

            "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.300,
            },

            "done": {
                "mode": "on_reward_override",
            }
        }
     },

    "basic-brain-discrete-1": {
        "type": "brain",

        "checkpoint": "/home/jake/robotai/_checkpoints/basic-brain-discrete-1-sb3-run21.zip",
        "load_fn": "src.models.stable_baselines3.load.load_stable_baselines3_dqn_actor",

        "models": {
            "vision": "yolov7-tiny-s53",
            "reward": "yolov7-tiny-prioritize_centered_nms",
        },

        "msgvec": {
            "obs": [
                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.vel",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.leftMotor.current",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.rightMotor.vel",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "odriveFeedback.rightMotor.current",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "voltage.volts",
                    "index": -1,
                    "timeout": 0.125,
                    "filter": {
                        "field": "voltage.type",
                        "op": "eq",
                        "value": "mainBattery",
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 15],
                        "vec_range": [-1, 1],
                    }
                },
                
                { 
                    "type": "msg",
                    "path": "headFeedback.pitchAngle",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.pitchMotorPower",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 1],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.yawAngle",
                    "index": -5,
                    "timing_index": -1,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-45.0, 45.0],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headFeedback.yawMotorPower",
                    "index": -3,
                    "timeout": 0.125,
                    "transform": {
                        "type": "rescale",
                        "msg_range": [0, 1],
                        "vec_range": [-1, 1],
                    },
                },

                # These messages provide feedback on the last few commands send to the motors, as there is some delay between a desired action and its effect
                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityLeft",
                    "index": -10,
                    "timeout": 0.15,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "index": -10,
                    "timeout": 0.15,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.pitchAngle",
                    "index": -10,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "headCommand.yawAngle",
                    "index": -10,
                    "timeout": 0.15,
                    "transform": {
                        "type": "rescale",
                        "vec_range": [-1, 1],
                        "msg_range": [-45.0, 45.0],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.0",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.1",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "accelerometer.acceleration.v.2",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "accelerometer.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-20, 20],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.0",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.1",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                { 
                    "type": "msg",
                    "path": "gyroscope.gyro.v.2",
                    "index": -30,
                    "timing_index": -1,
                    "timeout": 0.050,
                    "filter": {
                        "field": "gyroscope.sensor",
                        "op": "eq",
                        "value": 3, # 3 is the sensor on the simplebgc head
                    },
                    "transform": {
                        "type": "rescale",
                        "msg_range": [-2, 2],
                        "vec_range": [-1, 1],
                    },
                },

                {
                    "type": "vision",
                    "size": 17003,
                    "timeout": 0.100,
                    "index": [-1, -2],
                }
            ],

            "act": [
                {
                    "type": "discrete_msg",
                    "path": "odriveCommand.desiredVelocityLeft",
                    "initial": 0.0,
                    "range": [-1.0, 1.0],
                    "choices": [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10],
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                {
                    "type": "discrete_msg",
                    "path": "odriveCommand.desiredVelocityRight",
                    "initial": 0.0,
                    "range": [-1.0, 1.0],
                    "choices": [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10],
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "discrete_msg",
                    "path": "headCommand.pitchAngle",
                    "initial": 0.0,
                    "range": [-45.0, 45.0],
                    "choices": [-5.0, -1.0, 1.0, 5.0],
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },

                { 
                    "type": "discrete_msg",
                    "path": "headCommand.yawAngle",
                    "initial": 0.0,
                    "range": [-45.0, 45.0],
                    "choices": [-5.0, -1.0, 1.0, 5.0],
                    "timeout": 0.125,
                    "transform": {
                        "type": "identity",
                    },
                },
            ],

            "rew": {
                "base": "reward",

                "override": {
                    "positive_reward": 10.0,
                    "positive_reward_timeout": 2.0,

                    "negative_reward": -15.0,
                    "negative_reward_timeout": 2.0,
                }
            },

            "appcontrol": {
                "mode": "steering_override_v1",
                "timeout": 0.300,
            },

            "done": {
                "mode": "on_reward_override",
            }
        }
    }
})


WEB_CONFIG = dotdict({
   # Unknown
})

