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

from torchastic import Compass, StochasticAccumulator
import random

from aim import Run, Image as AimImage
from datetime import datetime
from PIL import Image as PILImage

from src.dataloaders.dataloader import TextImageDataset

from src.model.identity_ae import AutoEncoder

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
    act_fn: str
    resume_checkpoint: Optional[str]


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
    optimizer = Compass(
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
        betas=(0.9, 0.999),
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


def random_crop(image, crop_size=128):
    """Randomly crop a `crop_size x crop_size` region from the image."""
    _, _, h, w = image.shape
    top = torch.randint(0, h - crop_size + 1, (1,)).item()
    left = torch.randint(0, w - crop_size + 1, (1,)).item()
    return image[:, :, top : top + crop_size, left : left + crop_size]


def random_augment(image):
    """Apply the same random augmentation (90-degree rotation and flip) to both image and latent."""
    k = torch.randint(0, 4, (1,)).item()  # Random 90-degree rotation (0, 90, 180, 270 degrees)
    flip = torch.randint(0, 2, (1,)).item()  # Random horizontal flip

    image = torch.rot90(image, k, dims=[-2, -1])

    if flip:
        image = torch.flip(image, dims=[-1])

    return image


def train_ae(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config_identity.json")

    training_config = TrainingConfig(**config_data["training"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])

    # aim logging
    if training_config.aim_path is not None and rank == 0:
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run = Run(repo=training_config.aim_path, run_hash=training_config.aim_hash, experiment=training_config.aim_experiment_name)

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

    model = AutoEncoder(**asdict(model_config))
    if model_config.resume_checkpoint:
        model.load_state_dict(torch.load(model_config.resume_checkpoint))

    model.train()
    model.to(torch.bfloat16)
    model.to(rank)

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
                trained_layer_keywords = [n for n, _ in model.named_parameters()]

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
                # do this inside for loops!
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    images_cropped = random_crop(images[tmb_i * mb : tmb_i * mb + mb], crop_size=training_config.training_res)
                    images_aug = random_augment(images_cropped)
                    latent = model.encode(images_aug)

                    recon = model.decode(latent)

                    loss = F.l1_loss(
                        recon,
                        images_aug,
                    ) / (dataloader_config.batch_size / mb)

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
                            val_images_cropped = random_crop(val_images[tmb_i * mb : tmb_i * mb + mb], crop_size=256)
                            val_recon = model(val_images_cropped)
                            val_loss = F.mse_loss(
                                val_recon,
                                val_images_cropped,
                            ) / (dataloader_config.batch_size / mb)
                            val_loss_log.append(
                                val_loss * (dataloader_config.batch_size / mb)
                            )

                            # store first micro batch as preview if there's no preview image
                            if tmb_i == 0 and not preview:
                                preview["recon"] = val_recon.clone()
                                preview["ground_truth"] = val_images_cropped
                                grid_preview = torch.cat(
                                    [preview["recon"], preview["ground_truth"]], dim=0
                                )
                                grid = make_grid(
                                    grid_preview.clip(-1, 1),
                                    nrow=8,
                                    padding=2,
                                    normalize=True,
                                )
                                image_folder = f"{training_config.save_folder}/preview"
                                file_path = f"{image_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                                os.makedirs(image_folder, exist_ok=True)
                                save_image(
                                    grid,
                                    file_path,
                                )

                                # upload preview to aim

                                # Load your file_path into a PIL image if it isn't already
                                img_pil = PILImage.open(file_path)
                                # Wrap it for Aim
                                aim_img = AimImage(img_pil)
                                run.track(aim_img, name='example_image', step=global_step)

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
