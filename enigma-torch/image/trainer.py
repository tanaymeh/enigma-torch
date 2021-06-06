import os
import platform
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Union

from ..helpers.utils import logToUser, isValidDevice

class Trainer:
    def __init__(
        self,
        model,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: Optional[Union['torch.utils.data.DataLoader', None]] = None,
        device: Optional[str] = 'cpu',
        apex: Optional[bool] = False,
        progress: Optional[Union[bool, str]] = 'tqdm'
    ) -> None:

        """
        An object of type "Trainer" is the entry point to any training code

        Args:
            model: The Torch Model you want to train
            train_dataloader: Training Dataloader
            valid_dataloader (Optional): Validation Dataloader
                                         None defaults to not doing validation        
            device (Optional): Accelerator you want to use. 
                               None defaults to 'cpu' -- {'cuda' or 'cpu'}
            apex (Optional): Use NVIDIA's apex (torch.cuda.amp) or not
                             Defaults to False -- {True or False} 
            progress (Optional): Progress bar you want to use.
                                 Defaults to 'tqdm' -- {'tqdm' or 'rich'}
        """
        
        # Sanity checks and flags
        self.valid_flag = True if self.valid_dataloader else False
        
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), \
            "Train Dataloader must be of type Torch DataLoader"
        assert isinstance(valid_dataloader, torch.utils.data.DataLoader), \
            "Validation Dataloader must be of Torch DataLoader"
        assert device in ('cpu', 'cuda'), \
            "Device must be either 'cpu' or 'cuda'"
        assert apex in (True, False), \
            "Apex must be either True or False"
        assert progress in ('tqdm', 'rich'), \
            "Progress bar can be either 'tqdm' or 'rich'"
        
        self.model = model
        self.progress = progress
        self.train_dataloader = train_dataloader
        self._cuda_status = False
        
        self.valid_dataloader = valid_dataloader if self.valid_flag else None
        self.device = self._set_backend(device)
        self.apex = self._validate_apex(apex)
        
    @property
    def device(self):
        return self.device
    
    @device.setter
    def device(self, device):
        warnings.warn("Changing device explicitly can lead to some apex being disabled.")
        if isValidDevice(device):
            self.device = torch.device(device)
        else:
            raise ValueError(f"Device: {device} is not a valid device type. Using existing device choice.")
        
    def _set_backend(self, device):
        """Sets backend CPU or GPU backend for training and validation

        Args:
            device (str): Type of accelerator to use 
        """
        if device == "cuda" and torch.cuda.is_available():
            logToUser("Using Backend 'cuda:0'")
            logToUser(f"GPU found: {torch.cuda.get_device_name()}")
            self._cuda_status = True
            return torch.device("cuda:0")
        
        elif device == "cuda" and not torch.cuda.is_available():
            logToUser("Cannot set 'cuda' backend, falling back to 'cpu'")
            logToUser(f"CPU found: {platform.processor}")
            return torch.device("cpu")
        
        else:
            logToUser("Using Backend 'cpu'")
            logToUser(f"CPU found: {platform.processor}")
            return torch.device("cpu")
    
    def _validate_apex(self, apex):
        """Checks if apex can be enabled or not

        Args:
            apex (bool): True if apex is needed, False if not
        """
        if self._cuda_status and self.apex:
            return True
        return False