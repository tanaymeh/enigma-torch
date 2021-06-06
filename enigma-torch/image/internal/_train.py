import os
import sys
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trainRoutine(*args, **kwargs):
    """
    Main training function for image classification / regression
    """
    model = args[0]
    train_loader = args[1]
    train_loss_fn = args[2]
    optimizer = args[3]
    scheduler = args[4]
    device = args[5]
    apex_status = args[6]
    
    # Start the training procedure
    model.train()
    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    all_target = []
    all_pred = []
    for idx, (img, target) in prog_bar:
        img, target = img.to(device), target.to(device)
        
        