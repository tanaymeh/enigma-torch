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
        train_dataloader: 'EnigmaDataLoader',
        valid_dataloader: Optional['EngimaDataLoader']=None,

    ) -> None:
        """
        An object of type "Trainer" is the entry point to any training code

        Args:

        """