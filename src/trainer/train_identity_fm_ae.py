import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split

from torchastic import AdamW, StochasticAccumulator
import random

from aim import Run, Image as AimImage
from datetime import datetime
from PIL import Image as PILImage

from src.dataloaders.dataloader import TextImageDataset

from src.model.identity_ae import FlowmatchingAutoEncoder

import time


@dataclass
class TrainingConfig:
    master_seed: int
    train_minibatch: int
    lr: float
    weight_decay: float
    warmup_steps: int
    change_layer_every: int
    save_every: int
    save_folder: str
    validate_every: int
    training_res: int
    crop_count: int
    equivariance_enforcement_ratio: int
    force_resume: bool
    cfg_dropout_probability: float = 0.1
    aim_path: Optional[str] = None
    aim_experiment_name: Optional[str] = None
    aim_hash: Optional[str] = None
    aim_steps: Optional[int] = 0


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    resolution_step: int
    num_workers: int
    prefetch_factor: int
    ratio_cutoff: float
    thread_per_worker: int
    val_percentage: int


@dataclass
class ModelConfig:
    pixel_channels: str
    bottleneck_channels: str
    up_layer_blocks: list[list[int]]
    down_layer_blocks: list[list[int]]
    flow_matching_head_block: list[int]
    act_fn: str
    trainable_keywords: list[str]
    resume_checkpoint: Optional[str]
    cfg: float
    steps: str


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function

    # wipe the flag just in case
    for name, param in model.named_parameters():
        param.requires_grad = False
    trained_params = []
    for name, param in model.named_parameters():
        # Exclude 'norm_final.weight' from being included
        if "norm_final.weight" in name:
            param.requires_grad = False

        elif any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)
    # init optimizer
    optimizer = AdamW(
        [
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" not in name and "norm" not in name)
                ]
            },
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" in name or "norm" in name)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, hooks, trained_params


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def optimizer_state_to(optimizer, device):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            # Check if the item is a tensor
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def save_part(model, trained_layer_keywords, counter, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(
        filtered_state_dict, os.path.join(save_folder, f"trained_part_{counter}.pth")
    )


def save_config_to_json(filepath: str, **configs):
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def transform(image, latent):
    """
    Apply random equivariant transformations to tensor.
    
    Args:
        image: Input tensor [N, C, H, W]
        
    Returns:
        Transformed tensor [N, C, H', W'] (size may vary)
    """

    i_n, i_c, i_h, i_w = image.shape
    l_n, l_c, l_h, l_w = latent.shape
    
    compression_level = i_h / l_h
    # Discrete random scale (0.5image to 2image)
    scale = random.uniform(0.5, 2.0)
    if scale != 1.0:
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = int(round(new_h / compression_level) * compression_level)
        new_w = int(round(new_w / compression_level) * compression_level)

        mode = random.choice(['bilinear', 'bicubic', 'nearest'])
        image = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=False if mode != 'nearest' else None)
        latent = F.interpolate(latent, size=(int(new_h/compression_level), int(new_w/compression_level)), mode=mode, align_corners=False if mode != 'nearest' else None)
    
    # Random flip
    flip = random.choice(['h', 'v', 'none'])
    if flip == 'h':
        image = torch.flip(image, dims=[3])
        latent = torch.flip(latent, dims=[3])
        # print("flip h")
    elif flip == 'v':
        image = torch.flip(image, dims=[2])
        latent = torch.flip(latent, dims=[2])
        # print("flip v")
    
    # Random rotation (90° increments)
    rot = random.choice([0, 90, 180, 270])
    if rot == 90:
        image = torch.flip(image.transpose(-2, -1), dims=[-1])
        latent = torch.flip(latent.transpose(-2, -1), dims=[-1])
        # print("rot 90")
    elif rot == 180:
        image = torch.flip(torch.flip(image, dims=[-2]), dims=[-1])
        latent = torch.flip(torch.flip(latent, dims=[-2]), dims=[-1])
        # print("rot 180")
    elif rot == 270:
        image = torch.flip(image.transpose(-2, -1), dims=[-2])
        latent = torch.flip(latent.transpose(-2, -1), dims=[-2])
        # print("rot 270")
    
    return image, latent


def random_augment(image):
    """Apply the same random augmentation (90-degree rotation and flip) to both image and latent."""
    k = torch.randint(0, 4, (1,)).item()  # Random 90-degree rotation (0, 90, 180, 270 degrees)
    flip = torch.randint(0, 2, (1,)).item()  # Random horizontal flip

    image = torch.rot90(image, k, dims=[-2, -1])

    if flip:
        image = torch.flip(image, dims=[-1])

    return image

def random_augment(image, crop_size=320):
    """
    Apply the same random augmentation (random crop, 90-degree rotation, and flip) to the image.
    Handles images smaller than crop_size by padding them first.
    """
    image_shape = image.shape
    h, w = image_shape[-2], image_shape[-1]

    # Pad the image if it's smaller than the crop size
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h > 0 or pad_w > 0:
        # Pad on all sides (left, right, top, bottom)
        # The padding tuple is (pad_left, pad_right, pad_top, pad_bottom)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        image = F.pad(image, padding, "constant", 0)

    # Get the new height and width after padding
    padded_h, padded_w = image.shape[-2], image.shape[-1]

    # Get random crop coordinates
    top = torch.randint(0, padded_h - crop_size + 1, (1,)).item()
    left = torch.randint(0, padded_w - crop_size + 1, (1,)).item()

    # Perform the crop
    image = image[..., top:top+crop_size, left:left+crop_size]

    # Apply random rotation
    k = torch.randint(0, 4, (1,)).item()  # Random 90-degree rotation (0, 90, 180, 270 degrees)
    image = torch.rot90(image, k, dims=[-2, -1])

    # Apply random horizontal flip
    if torch.randint(0, 2, (1,)).item():
        image = torch.flip(image, dims=[-1])

    return image



def train_ae(rank, world_size, debug=False, json_config_file="training_config_flow_match.json"):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json(json_config_file)

    training_config = TrainingConfig(**config_data["training"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])

    # aim logging
    if training_config.aim_path is not None and rank == 0:
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = Run(repo=training_config.aim_path, run_hash=training_config.aim_hash, experiment=training_config.aim_experiment_name, force_resume=training_config.force_resume)

        hparams = config_data.copy()
        hparams["training"]['aim_path'] = None
        run["hparams"] = hparams

    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )
    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    model = FlowmatchingAutoEncoder(**asdict(model_config))
    if model_config.resume_checkpoint:
        state_dict = torch.load(model_config.resume_checkpoint)
        model.load_state_dict(state_dict, strict=False) # fm head will be inited from scratch if there isn't any

    model.train()
    model.to(torch.bfloat16)
    model.to(rank)
    
    params_to_optimize = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in model_config.trainable_keywords):
            params_to_optimize.append(name)
            print("these weights are trained", name)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        base_res=dataloader_config.base_resolution,
        shuffle_tags=False,
        tag_drop_percentage=0.0,
        uncond_percentage=0.0,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
    )
    # split train validation set
    train_set_size = int(len(dataset) * (1 - dataloader_config.val_percentage))
    val_set_size = len(dataset) - train_set_size
    dummy_collate_fn = dataset.dummy_collate_fn
    dataset, val_dataset = random_split(dataset, [train_set_size, val_set_size])

    # validation set
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,  # batch size is handled in the dataset
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        prefetch_factor=dataloader_config.prefetch_factor,
        pin_memory=True,
        collate_fn=dummy_collate_fn,
    )

    optimizer = None
    scheduler = None
    hooks = []
    optimizer_counter = 0
    global_step = training_config.aim_steps
    while True:
        training_config.master_seed += 1
        torch.manual_seed(training_config.master_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is handled in the dataset
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor,
            pin_memory=True,
            collate_fn=dummy_collate_fn,
        )
        for counter, data in tqdm(
            enumerate(dataloader),
            total=len(dataset),
            desc=f"training, Rank {rank}",
            position=rank,
        ):
            images, _, _ = data[0]

            images = images.to(rank, non_blocking=True)

            if counter % training_config.change_layer_every == 0:
                # periodically remove the optimizer and swap it with new one
                trained_layer_keywords = params_to_optimize

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                optimizer, scheduler, hooks, trained_params = init_optimizer(
                    model,
                    trained_layer_keywords,
                    training_config.lr,
                    training_config.weight_decay,
                    training_config.warmup_steps,
                )

            optimizer_counter += 1

            # aliasing
            mb = training_config.train_minibatch
            loss_log = []
            for tmb_i in tqdm(
                range(dataloader_config.batch_size // mb // world_size),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    shunt_model = torch.rand(()) > 0.5
                    images_aug = []
                    for x in range(training_config.crop_count):
                        images_aug.append(random_augment(images[tmb_i * mb : tmb_i * mb + mb], training_config.training_res))
                    images_aug = torch.cat(images_aug, dim=0)
                    # --- Start of Flow Matching Loss Modification ---

                    # Generate conditioning signal from the autoencoder.
                    # Since the AE is pretrained, we run this in no_grad mode to save memory and compute,
                    # and prevent its weights from being updated.
                    # NOTE: gemini, we need to set this to be trainable too later on, i think it will help creating better latents
                    with torch.no_grad():
                        latent = model.encode(images_aug, skip_last_downscale=shunt_model)
                        # The decoder's output before the final conv layer acts as conditioning
                        conditioning_logits = model.decode(latent, skip_second_upscale=shunt_model, logits_only=True)

                    # Classifier-Free Guidance: randomly drop conditioning to train an unconditional model
                    # If a random number is less than the dropout probability, we use zero tensors for conditioning.
                    if torch.rand(()) < getattr(training_config, 'cfg_dropout_probability', 0.1):
                        conditioning_logits = torch.zeros_like(conditioning_logits)

                    # 1. Sample timesteps 't' from the custom distribution
                    x_dist, prob_dist = model.create_distribution(1000, device=rank)
                    t = model.sample_from_distribution(x_dist, prob_dist, (images_aug.shape[0],), device=rank).view(-1, 1, 1, 1)

                    # 2. Sample noise from a standard normal distribution
                    noise = torch.randn_like(images_aug)

                    # 3. Create the noisy image x_t by interpolating between the true image and noise
                    # Path: x_t = (1-t) * image + t * noise
                    xt = (1.0 - t) * images_aug + t * noise

                    # 4. Get the model's prediction of the vector field
                    predicted_velocity = model.flow_matching_decode(xt, conditioning_logits)

                    # 5. Define the target vector field for the flow
                    target_velocity = model.fm_target(noise, images_aug)  # This is (noise - image)

                    # 6. Calculate the flow matching loss
                    loss = F.mse_loss(
                        predicted_velocity,
                        target_velocity,
                    ) / (dataloader_config.batch_size / mb)
                    
                    # --- End of Flow Matching Loss Modification ---

                loss.backward()
                loss_log.append(
                    loss.detach().clone() * (dataloader_config.batch_size / mb)
                )
            loss_log = sum(loss_log) / len(loss_log)

            optimizer_state_to(optimizer, rank)
            StochasticAccumulator.reassign_grad_buffer(model)

            if not debug:
                synchronize_gradients(model)

            scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            if rank == 0:
                run.track(loss_log, name='loss', step=global_step)
                run.track(training_config.lr, name='learning_rate', step=global_step)

            optimizer_state_to(optimizer, "cpu")
            torch.cuda.empty_cache()

            # do validation loss
            if counter % training_config.validate_every == 0 and rank == 0:

                # store validation loss
                val_loss_log = []
                preview = {}
                # loop through all validation set and get the validation loss and first mini batch sample from rank 0
                for val_data in tqdm(
                    val_dataloader,
                    total=len(val_dataset),
                    desc=f"validating, Rank {rank}",
                    position=rank,
                ):
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):

                        val_images, _, _ = val_data[0]
                        val_images = val_images.to(rank, non_blocking=True)
                        mb = training_config.train_minibatch

                        for tmb_i in tqdm(
                            range(dataloader_config.batch_size // mb // world_size),
                            desc=f"minibatch validation, Rank {rank}",
                            position=rank,
                        ):
                            val_images_cropped = val_images[0 : max(training_config.train_minibatch//8, 1)]
                            
                            # --- Start of Validation Modification ---

                            # 1. Get standard AE reconstruction for comparison
                            val_recon_ae = model.decode(model.encode(val_images_cropped), logits_only=False)
                            val_recon_ae_shunt = model.decode(model.encode(val_images_cropped, skip_last_downscale=True), skip_second_upscale=True,  logits_only=False)


                            # 2. Generate image using the Flow Matching head
                            # Get conditioning features from the pretrained AE
                            conditioning_logits = model.decode(model.encode(val_images_cropped), logits_only=True)
                            conditioning_logits_shunt = model.decode(model.encode(val_images_cropped, skip_last_downscale=True), skip_second_upscale=True, logits_only=True)
                            
                            # Define a simple wrapper for the solver, which expects a callable model
                            solver_model = lambda x, guidance_logits: model.flow_matching_decode(x, guidance_logits)
                            
                            # Prepare for the ODE solving process
                            timesteps = torch.linspace(1, 0, model_config.steps, device=rank).tolist() # 50 steps for validation
                            initial_noise = torch.randn_like(val_images_cropped)

                            # Generate image by solving the ODE from t=1 (noise) to t=0 (image)
                            generated_image = FlowmatchingAutoEncoder.denoise_cfg(
                                model=solver_model,
                                timesteps=timesteps,
                                img=initial_noise,
                                cond=conditioning_logits,
                                neg_cond=torch.zeros_like(conditioning_logits),
                                cfg=model_config.cfg, # CFG scale
                                first_n_steps_without_cfg=0 # Use CFG after a few steps
                            )

                            generated_image_shunt = FlowmatchingAutoEncoder.denoise_cfg(
                                model=solver_model,
                                timesteps=timesteps,
                                img=initial_noise,
                                cond=conditioning_logits_shunt,
                                neg_cond=torch.zeros_like(conditioning_logits_shunt),
                                cfg=model_config.cfg, # CFG scale
                                first_n_steps_without_cfg=0 # Use CFG after a few steps
                            )

                            # Calculate validation loss between generated and ground truth images
                            val_loss = F.mse_loss(generated_image, val_images_cropped)
                            val_loss_log.append(val_loss.item())

                            # store first micro batch as preview if there's no preview image
                            if tmb_i == 0 and not preview:
                                preview["generated"] = generated_image.clone()
                                preview["recon_ae"] = val_recon_ae.clone()
                                preview["generated_shunt"] = generated_image_shunt.clone()
                                preview["recon_ae_shunt"] = val_recon_ae_shunt.clone()
                                preview["ground_truth"] = val_images_cropped.clone()
                                
                                # Create a grid for visual comparison
                                grid_preview = torch.cat(
                                    [preview["ground_truth"], preview["recon_ae"], preview["recon_ae_shunt"], preview["generated"], preview["generated_shunt"]], dim=0
                                )
                                grid = make_grid(
                                    grid_preview.clip(-1, 1),
                                    nrow=val_images_cropped.shape[0], # Show one row per category
                                    padding=2,
                                    normalize=True,
                                )
                                image_folder = f"{training_config.save_folder}/preview"
                                file_path = f"{image_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                                os.makedirs(image_folder, exist_ok=True)
                                save_image(
                                    grid,
                                    file_path,
                                )

                                # upload preview to aim
                                img_pil = PILImage.open(file_path)
                                aim_img = AimImage(img_pil)
                                run.track(aim_img, name='example_image', step=global_step)
                                break
                            break
                                
                            # --- End of Validation Modification ---

                # aggregate the loss
                with torch.no_grad():
                    val_loss_log = sum(val_loss_log) / len(val_loss_log)

            if (counter + 1) % training_config.save_every == 0 and rank == 0:
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(
                    model.state_dict(),
                    model_filename,
                )

            if not debug:
                dist.barrier()
            
            global_step += 1

        # save final model
        if rank == 0:
            model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            torch.save(
                model.state_dict(),
                model_filename,
            )

    if not debug:
        dist.destroy_process_group()
