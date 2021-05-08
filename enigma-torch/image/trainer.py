import os
import platform
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Union


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: Optional['torch.utils.data.DataLoader'] = None,
        accelerator: Optional[str] = 'cpu',
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
            accelerator (Optional): Accelerator you want to use. 
                                    None defaults to 'cpu' -- {'cuda' or 'cpu'}
            apex (Optional): Use NVIDIA's apex (torch.cuda.amp) or not
                             Defaults to False -- {True or False} 
            progress (Optional): Progress bar you want to use.
                                 Defaults to 'tqdm' -- {'tqdm' or 'rich'}
        """
        
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), \
            "Train Dataloader must be a Torch DataLoader"
        assert accelerator in ('cpu', 'cuda'), \
            "Accelerator must be either 'cpu' or 'cuda'"
        assert 
        
        self.model = model
        self.train_dataloader = train_dataloader
        