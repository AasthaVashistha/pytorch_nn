import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Model Hyperparameters
mb_size = 2
epochs = 100
learning_rate = 1e-2

# Loss Function and Optimizer
bce_loss = nn.BCELoss()
