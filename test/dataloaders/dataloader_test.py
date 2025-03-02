import sys
import os
from torchvision.utils import save_image, make_grid
import json
# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

from src.dataloaders.dataloader import TextImageDataset


dataset = TextImageDataset(
    batch_size=64,
    jsonl_path="tags_data.jsonl",
    image_folder_path="",
    # tag_implication_path="furry_50k_4o/tag_implications-2024-06-16.csv",
    base_res=[
        512,
    ],  # resolution computation should be in here instead of in csv prep!
    shuffle_tags=True,
    tag_drop_percentage=0.0,
    uncond_percentage=0.1,
    resolution_step=32,
    seed=0,
    rank=0,
    num_gpus=1,
    ratio_cutoff=2,
)

os.makedirs("tag_preview", exist_ok=True)

for i in tqdm(range(10)):
    images, caption, index = dataset[i]
    with open(f"tag_preview/{i}.jsonl", 'w') as f:
        for item in caption:
            json.dump(item, f)  # Dump the item as a JSON object
            f.write('\n')  # Write a newline after each JSON object
    save_image(make_grid(images.clip(-1, 1)), f"tag_preview/{i}.jpg", normalize=True)

print()
