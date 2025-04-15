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

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    # 각 t에 대해 sqrt(alpha_bar)와 sqrt(1 - alpha_bar) 계산
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

# Device 설정
if __name__ == "__main__":
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    T_steps = 1000
    beta_start = 1e-5
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T_steps).to(device)  # (T_steps,)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 누적곱, (T_steps,)

    path_HR = f"data/Sun-Hays80/image_SRF_{resolution}/img_00{i}_SRF_{resolution}_HR.png"


    model = UNet_small().to(device)

    x = torch.rand(1,6,256,256, device=device)


    T_steps = 1000
    t = torch.randint(0, T_steps, (1,), device=device).long()
    y_hat = model(x, t)
    breakpoint()

