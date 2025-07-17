import sys
import os

# Add the project root to `sys.path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
import torch
from src.model.identity_ae import CEAutoEncoder

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid
from torchvision.utils import save_image
from typing import Optional

def resize_image(img, target_size, is_height=False):
    """
    Resize an image while preserving aspect ratio.

    Args:
        img (str): pil image
        output_path (str): Path to save the resized image
        target_size (int): Target width or height in pixels
        is_height (bool): If True, target_size is height; otherwise, it's width
    """
    # Open the image

    # Get current dimensions
    width, height = img.size

    # Calculate new dimensions
    if is_height:
        # If target is height, calculate new width
        new_height = target_size
        new_width = int(width * (new_height / height))
    else:
        # If target is width, calculate new height
        new_width = target_size
        new_height = int(height * (new_width / width))

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img


def center_crop_to_divisible(image, block_size):
    """
    Center crops an image tensor to dimensions that are divisible by block_size.

    Args:
        image: A tensor of shape [C, H, W]
        block_size: The block size to make dimensions divisible by

    Returns:
        A center-cropped tensor with dimensions divisible by block_size
    """
    _, h, w = image.shape

    # Calculate new dimensions divisible by block_size
    new_h = h - (h % block_size)
    new_w = w - (w % block_size)

    # Calculate crop margins
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    # Perform center crop
    cropped_image = image[:, top : top + new_h, left : left + new_w]

    return cropped_image


def benchmark_encoding_decoding(ae, image_tensor, device="cuda:0", num_runs=5, save_outputs=True, bypass=False):
    """
    Benchmark the encoding and decoding times of an autoencoder model.
    
    Args:
        ae: The autoencoder model
        image_tensor: The input image tensor
        device: Device to run the model on
        num_runs: Number of runs to average timing over
        save_outputs: Whether to save output images
        
    Returns:
        dict: Dictionary containing timing results
    """
    # Ensure model is in eval mode
    ae.eval()
    
    # Move image to device
    input_image = image_tensor.unsqueeze(0).to(device) * 2 - 1
    
    # Store timing results
    encode_times = []
    decode_times = []
    total_times = []
    
    # Warm-up run
    print("Performing warm-up run...")
    with torch.no_grad():
        _ = ae.encode(input_image, checkpoint=True, skip_last_downscale=torch.tensor(False))
        _ = ae.decode(_, checkpoint=True, skip_second_upscale=torch.tensor(False))
    
    # Benchmark runs
    print(f"Performing {num_runs} benchmark runs...")
    for i in range(num_runs):
        with torch.no_grad():
            # Clear CUDA cache to ensure fair benchmarking
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            # Measure encoding time
            encode_start = time()
            latent = ae.encode(input_image, checkpoint=True, skip_last_downscale=torch.tensor(bypass))
            torch.cuda.synchronize() if device.startswith("cuda") else None
            encode_end = time()
            
            # Measure decoding time
            decode_start = time()
            image_recon = ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(bypass))
            torch.cuda.synchronize() if device.startswith("cuda") else None
            decode_end = time()
            
            # Normalize the reconstructed image
            image_recon = torch.clip((image_recon + 1) / 2, 0, 1)
            
            # Calculate times
            encode_time = encode_end - encode_start
            decode_time = decode_end - decode_start
            total_time = decode_end - encode_start
            
            encode_times.append(encode_time)
            decode_times.append(decode_time)
            total_times.append(total_time)
            
            print(f"Run {i+1}: Encode: {encode_time:.4f}s, Decode: {decode_time:.4f}s, Total: {total_time:.4f}s")
    
    # Calculate statistics
    encode_mean = np.mean(encode_times)
    encode_std = np.std(encode_times)
    decode_mean = np.mean(decode_times)
    decode_std = np.std(decode_times)
    total_mean = np.mean(total_times)
    total_std = np.std(total_times)
    
    # Print results
    print("\n===== BENCHMARK RESULTS =====")
    print(f"Encoding time: {encode_mean:.4f}s ± {encode_std:.4f}s")
    print(f"Decoding time: {decode_mean:.4f}s ± {decode_std:.4f}s")
    print(f"Total time: {total_mean:.4f}s ± {total_std:.4f}s")
    
    # Save outputs from the last run
    if save_outputs:
        # Create and save latent visualization
        grid = make_grid(latent.permute(1, 0, 2, 3), nrow=8, padding=2, normalize=True)
        save_image(grid, "latent_benchmark.png")
        
        # Save reconstructed image
        save_image(image_recon, "recon_benchmark.png")
        
        print("Saved output images as 'latent_benchmark.png' and 'recon_benchmark.png'")
    
    return {
        "encode_time": {"mean": encode_mean, "std": encode_std},
        "decode_time": {"mean": decode_mean, "std": decode_std},
        "total_time": {"mean": total_mean, "std": total_std}
    }


def create_grouped_rgb_visualization(latent, pad=2, normalize=True):
    """
    Groups the channels of a latent tensor into sets of 3 for RGB visualization.
    For 32 channels, this will create 11 RGB images (with the last one padded with zeros).
    
    Args:
        latent: Tensor of shape [batch, channel, height, width]
        pad: Padding between images in the grid
        normalize: Whether to normalize each RGB group independently
        
    Returns:
        Grid of RGB visualizations
    """
    # Remove batch dimension if present
    if latent.dim() == 4:
        latent = latent.squeeze(0)
    
    c, h, w = latent.shape
    
    # Calculate how many complete RGB triplets we can make
    num_complete_triplets = c // 3
    
    # Calculate remaining channels that need padding
    remaining_channels = c % 3
    
    # Create a list to hold our RGB groupings
    rgb_groups = []
    
    # Group channels in sets of 3 for RGB
    for i in range(num_complete_triplets):
        # Extract 3 consecutive channels
        rgb_group = latent[i*3:(i+1)*3]
        
        # Normalize this group independently if requested
        if normalize:
            rgb_group = (rgb_group - rgb_group.min()) / (rgb_group.max() - rgb_group.min() + 1e-8)
        
        rgb_groups.append(rgb_group)
    
    # Handle the remaining channels with padding
    if remaining_channels > 0:
        # Extract remaining channels
        last_group = latent[num_complete_triplets*3:]
        
        # Create a zero tensor with 3 channels
        padded_group = torch.zeros(3, h, w, device=latent.device)
        
        # Copy the remaining channels
        padded_group[:remaining_channels] = last_group
        
        # Normalize if requested
        if normalize:
            # Only normalize if we have non-zero values
            if padded_group.max() > padded_group.min():
                padded_group = (padded_group - padded_group.min()) / (padded_group.max() - padded_group.min() + 1e-8)
        
        rgb_groups.append(padded_group)
    
    # Create a grid of all RGB groupings
    grid = make_grid(rgb_groups, nrow=4, padding=pad, normalize=False)
    
    return grid, rgb_groups

def sample_from_logits(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
    """
    Applies temperature, top-k, and top-p (nucleus) sampling to logits to generate a sample.
    
    Args:
        logits (torch.Tensor): The raw output from the model. Shape: (N, 768, H, W)
        temperature (float): Controls randomness. Higher values ( > 1.0) make results more random,
                             lower values ( < 1.0) make them more deterministic. Must be > 0.
        top_k (int, optional): If set, filters the logits to only the top 'k' most likely values.
        top_p (float, optional): If set, filters the logits using nucleus sampling, keeping the smallest
                                 set of values whose cumulative probability exceeds 'p'.

    Returns:
        torch.Tensor: A tensor representing the sampled image with integer values [0, 255].
                      Shape: (N, 3, H, W)
    """
    N, _, H, W = logits.shape
    
    # Reshape for sampling: (N, 768, H, W) -> (N*3*H*W, 256)
    # This treats every pixel-channel as an independent classification problem.
    logits_for_sampling = logits.reshape(N, 3, 256, H, W).permute(0, 1, 3, 4, 2).reshape(-1, 256)

    # 1. Apply Temperature
    # A temperature of 0 would cause division by zero, so we use a very small value for determinism.
    if temperature == 0:
        # This is equivalent to argmax
        return torch.argmax(logits_for_sampling, dim=-1).reshape(N, 3, H, W)
    
    logits_for_sampling = logits_for_sampling / temperature

    # 2. Apply Top-k filtering
    if top_k is not None and top_k > 0:
        # Get the top k values and their indices
        top_k_vals, _ = torch.topk(logits_for_sampling, k=top_k, dim=-1)
        # The value of the k-th element becomes the threshold
        kth_vals = top_k_vals[:, -1].unsqueeze(dim=-1)
        # Set all logits lower than the k-th value to -inf
        indices_to_remove = logits_for_sampling < kth_vals
        logits_for_sampling[indices_to_remove] = -float("Inf")

    # 3. Apply Top-p (nucleus) filtering
    if top_p is not None and top_p > 0.0:
        # Convert logits to probabilities
        probs = F.softmax(logits_for_sampling, dim=-1)
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask for values to remove.
        # We remove tokens that are part of the tail of the distribution.
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask to the right to ensure we keep the first token that exceeds p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0 # Always keep at least the most likely token

        # Scatter the mask back to the original order of indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits_for_sampling[indices_to_remove] = -float("Inf")

    # 4. Sample from the final modified distribution
    final_probs = F.softmax(logits_for_sampling, dim=-1)
    sampled_indices = torch.multinomial(final_probs, num_samples=1)
    
    # Reshape back to image format (N, 3, H, W) and return
    return sampled_indices.reshape(N, 3, H, W)

if __name__ == "__main__":

    ae_args = {
        "pixel_channels": 3,
        "bottleneck_channels": 64,
        "down_layer_blocks": [[32, 10], [64, 15], [128, 20], [256, 20], [512, 20], [512, 20]],
        "up_layer_blocks": [[512, 20], [512, 20], [256, 20], [128, 20], [64, 15], [32, 10]], 
        "act_fn": "silu",
    }

    ae = CEAutoEncoder(**ae_args)
    state_dict = torch.load("identity_ae_64_cross_entrophy_rgb_space/2025-07-17_06-10-19.pth")

    # state_dict["encoder.out_conv.weight"] = torch.cat([state_dict["encoder.out_conv.weight"], 1/10 * torch.randn_like(state_dict["encoder.out_conv.weight"])])
    # state_dict["encoder.out_conv.bias"] = torch.cat([state_dict["encoder.out_conv.bias"], 1/10 * torch.randn_like(state_dict["encoder.out_conv.bias"])])
    # state_dict["decoder.in_conv.weight"] = torch.cat([state_dict["decoder.in_conv.weight"], 1/10 * torch.randn_like(state_dict["decoder.in_conv.weight"])], dim=1)

    # # Strip the _orig_mod. prefix from all keys
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     new_key = k.replace('_orig_mod.', '')
    #     new_state_dict[new_key] = v
    ae.load_state_dict(state_dict)
    ae.to(torch.bfloat16)
    ae.to("cuda:0")


    sampling_temp = 0.9
    sampling_top_k = 255
    sampling_top_p = 0.9

    # torch.save(ae.state_dict(), "64_ae.pth")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        resize = int(1024*1)
        block_size = 32
        # f = 7
        for f in range(5, 8):
            image = Image.open(f"recon{f}orig.jpg")
            image = resize_image(image, resize, is_height=False)
            image_tensor = to_tensor(image)
            image_tensor = center_crop_to_divisible(image_tensor, block_size)

            from time import time
        
            with torch.no_grad():
                start = time()
                latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(False))
                # latent = F.interpolate(latent, scale_factor=2, mode="nearest")
                stop1 = time()
                image_logits = ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(False))
                image_recon = sample_from_logits(image_logits, temperature=sampling_temp, top_k=sampling_top_k, top_p=sampling_top_p)
                
                # Normalize integer images [0, 255] to float [-1, 1] for saving/viewing
                image_recon = (image_recon.float() / 255.0)
                stop2 = time()
                grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                save_image(grid, f"latent{f}_.png")
                save_image(grid_rgb, f"latent{f}_rgb.png")
                save_image(image_recon, f"recon{f}_.png")
                save_image(image_tensor, f"recon{f}_orig.png")
                # image_recon = to_pil_image(image_recon.squeeze(0))
                # latent_grid = to_pil_image(grid)
                # latent_grid.save("latent.png")
                # image_recon.save("recon.png")
                print(stop1-start, stop2-start)

                start = time()
                latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # latent = F.interpolate(latent, scale_factor=2, mode="nearest")
                stop1 = time()
                image_logits = ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True))
                image_recon = sample_from_logits(image_logits, temperature=sampling_temp, top_k=sampling_top_k, top_p=sampling_top_p)
                
                # Normalize integer images [0, 255] to float [-1, 1] for saving/viewing
                image_recon = (image_recon.float() / 255.0)
                stop2 = time()
                grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                save_image(grid, f"latent{f}_shunt.png")
                save_image(grid_rgb, f"latent{f}_rgbshunt.png")
                save_image(image_recon, f"recon{f}_shunt.png")
                # image_recon = to_pil_image(image_recon.squeeze(0))
                # latent_grid = to_pil_image(grid)
                # latent_grid.save("latent.png")
                # image_recon.save("recon.png")
                print(stop1-start, stop2-start)

                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(False))
                # latent = F.interpolate(latent, scale_factor=2, mode="bicubic")
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(False)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_eq.png")
                # save_image(grid_rgb, f"latent{f}_rgbeq.png")
                # save_image(image_recon, f"recon{f}_eq.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)

                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # latent = torch.flip(latent, dims=[3])
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_flip.png")
                # save_image(grid_rgb, f"latent{f}_rgbflip.png")
                # save_image(image_recon, f"recon{f}_flip.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)

                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # latent = torch.flip(torch.flip(latent, dims=[-2]), dims=[-1])
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_rotate_180.png")
                # save_image(grid_rgb, f"latent{f}_rgb_rotate_180.png")
                # save_image(image_recon, f"recon{f}_rotate_180.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)

                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # latent = torch.flip(latent.transpose(-2, -1), dims=[-1])
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_rotate_270.png")
                # save_image(grid_rgb, f"latent{f}_rgb_rotate_270.png")
                # save_image(image_recon, f"recon{f}_rotate_270.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)


                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # latent = torch.flip(latent.transpose(-2, -1), dims=[-2])
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_rotate_90.png")
                # save_image(grid_rgb, f"latent{f}_rgb_rotate_90.png")
                # save_image(image_recon, f"recon{f}_rotate_90.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)

                # latent = ae.encode(image_tensor.unsqueeze(0).to("cuda:0") *2 -1, checkpoint=True, skip_last_downscale=torch.tensor(True))
                # # latent = torch.flip(latent.transpose(-2, -1), dims=[-2])
                # stop1 = time()
                # image_recon = torch.clip((ae.decode(latent, checkpoint=True, skip_second_upscale=torch.tensor(True)) + 1) /2, 0, 1)
                # stop2 = time()
                # grid = make_grid(latent.permute(1,0,2,3), nrow=8, padding=2, normalize=True)
                # grid_rgb, rgb_groups = create_grouped_rgb_visualization(latent)
                # save_image(grid, f"latent{f}_48.png")
                # save_image(grid_rgb, f"latent{f}_rgb_48.png")
                # save_image(image_recon, f"recon{f}_48.png")
                # # image_recon = to_pil_image(image_recon.squeeze(0))
                # # latent_grid = to_pil_image(grid)
                # # latent_grid.save("latent.png")
                # # image_recon.save("recon.png")
                # print(stop1-start, stop2-start)


                # results = benchmark_encoding_decoding(ae, image_tensor, save_outputs=False, num_runs=50)
                # results = benchmark_encoding_decoding(ae, image_tensor, save_outputs=False, num_runs=50, bypass=True)
        print()
