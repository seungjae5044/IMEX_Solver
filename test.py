import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import math
from tqdm.auto import tqdm
import numpy as np
from models import UNet, UNet_small

# Device 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

model = UNet_small().to(device)

x = torch.rand(1,6,256,256, device=device)


T_steps = 1000
t = torch.randint(0, T_steps, (1,), device=device).long()
y_hat = model(x, t)
breakpoint()

