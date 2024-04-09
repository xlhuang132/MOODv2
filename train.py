 
import torch 
import argparse
from config.defaults import update_config,_C as cfg
from trainer.build_trainer import build_trainer 
import os
import numpy as np 
def parse_args():
    parser = argparse.ArgumentParser(description="codes for MOOD")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/cifar10_supervised.yaml",
        type=str,
    ) 
    
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args) 
    trainer=build_trainer(cfg)
    trainer.train()
        
    
   