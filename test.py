import os
import platform
import warnings
import numpy as np

import torch
from torch.cuda import CudaError
import torch.nn as nn
import torch.nn.functional as F

print(np.__version__)
print(torch.__version__)
print(torch.cuda.is_available())