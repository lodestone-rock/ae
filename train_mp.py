import torch.multiprocessing as mp
import os
import torch

from src.trainer.train_ae import train_ae
if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_ae,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )   