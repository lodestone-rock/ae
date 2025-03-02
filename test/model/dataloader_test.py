import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
import torch
from src.model.ae import AutoEncoder



if __name__ == "__main__":

    x = torch.randn(16, 3, 256, 256).to("cuda:0")

    ae = AutoEncoder(
        3,
        4,
        down_layer_blocks=((32, 15), (64, 20), (96, 20), (128, 20)),
        up_layer_blocks=((128, 20), (96, 20), (64, 20), (32, 15)),
    )
    ae.to("cuda:0")
    recon = ae.forward(x, checkpoint=True)
    print()
