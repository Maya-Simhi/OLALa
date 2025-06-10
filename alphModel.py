import random
import quantizer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import torch.nn.functional as F



# PyTorch MLP Class remains the same
class AlphhMLP(torch.nn.Module):
    def __init__(self, input_size=6422):
        super(AlphhMLP, self).__init__()

        # MLP to transform 1D vector to CNN input
        self.fc1 = nn.Linear(input_size, 64 * 8 * 8)  
        self.relu = nn.ReLU()

        # CNN Layers
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers (Output 1 Number)
        self.fc2 = nn.Linear(64 , 128)  
        self.fc3 = nn.Linear(128, 1)  # Outputs a single number

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(-1, 64, 8, 8)  # Reshape to CNN format

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)  # Flatten

        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Final layer outputs 1 number

        return x  # No activation (for regression)



