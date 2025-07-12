import torch.multiprocessing as mp
import os
import torch

from src.trainer.train_identity_ae_refinement import train_ae
if __name__ == "__main__":
    
    # world_size = torch.cuda.device_count()
    train_ae(0, 1, True)
    # Use spawn method for starting processes
    # mp.spawn(
    #     train_ae,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True
    # )   