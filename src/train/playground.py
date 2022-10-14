import json

import torch

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.msgvec.pymsgvec import PyMsgVec

input_model = MODEL_CONFIGS[HOST_CONFIG.DEFAULT_BRAIN_CONFIG]
msgvec = PyMsgVec(json.dumps(input_model["msgvec"]).encode("utf-8"))
act_shape = msgvec.act_size() # scalar (default: 4)
obs_shape = msgvec.obs_size() # scalar (default: 17006)
pass
