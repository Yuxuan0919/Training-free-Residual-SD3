import torch
from pipeline_taca_flux import FluxPipeline

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
pipe = FluxPipeline.from_pretrained("/inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/flux/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

prompt = "A beautiful landscape with mountains and a river."

# Comment the following line if you just want training-free inference
pipe.load_lora_weights('/inspire/hdd/project/chineseculture/public/yuxuan/TACA/TACA/flux-dev-lora-rank-16.safetensors')

set_seed(42)
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=3.5
).images[0]

image.save("out-r16.png")   