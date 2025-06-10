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
import copy
import dataGetUtils
import gc


# PyTorch MLP Class remains the same
class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_hidden1=700, num_hidden2=900, num_hidden3=700, output_dim=2):
        super(PyTorchMLP, self).__init__()
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(200, num_hidden1)
        self.layer2 = torch.nn.Linear(num_hidden1, num_hidden2)
        self.layer3 = torch.nn.Linear(num_hidden2, num_hidden3)
        self.layer4 = torch.nn.Linear(num_hidden3, 800)
        self.layer5 = torch.nn.Linear(800, output_dim ** 2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = torch.tanh

    def forward(self, inp):
        inp = inp.reshape([-1, 200])
        first_layer = self.relu(self.layer1(inp))
        second_layer = self.relu(self.layer2(first_layer))
        third_layer = self.relu(self.layer3(second_layer))
        forth_layer = self.relu(self.layer4(third_layer))
        fith_layer = self.sigmoid(self.layer5(forth_layer))
        return torch.reshape(fith_layer, [self.output_dim, self.output_dim])


def train_model(model,model_alpha, args, criterion , learning_rate=1e-6, testData = [], optimizer = torch.nn.MSELoss(), train_loader = None,FL_model = None, anotherModel = None ):
    model.train()
    gettingAlph = None

    local_weights_orig = testData
    if(torch.all(local_weights_orig.eq(0))):
        return;
    hex_mat = model(torch.ones(200))
    mechanism = quantizer.LatticeQuantization(args, hex_mat, True)
    if(args.train_with_alpha):
        model_alpha.train()
        local_weights_orig = local_weights_orig.to(args.device)
        gettingAlph = model_alpha(local_weights_orig).to(args.device)
        reconstructed_points, vec = mechanism(input =local_weights_orig,gettingAlph = gettingAlph, shouldPrint =False, shouldReturnBack = True)
    else:
        # Lattice quantization with generated matrix
        reconstructed_points, vec = mechanism(input =local_weights_orig, shouldPrint =False, shouldReturnBack = True)
    
    # Compute loss and backpropagate
    if args.loss_by == "accuracy":
        test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')
        new_model = anotherModel
        _, shapes = dataGetUtils.getWeights(anotherModel)
        new_model = dataGetUtils.restoreWeights(shapes, reconstructed_points ,  new_model)
        val_acc, val_loss = dataGetUtils.test(train_loader, new_model, test_creterion, args.device, shouldKeppGrads = True)
        gc.collect()  # Free CPU memory
        torch.cuda.empty_cache()  # Free GPU memory

        loss = val_loss
        if loss<0:
            print('loss smaller then 0')
            loss = loss * -1
        print(f'loss for matrix train {loss}, quantizer loss {val_loss},{val_acc} ')
    elif args.loss_by == "snr":
        signal_power = torch.var(vec, unbiased=False)  # Power of the signal
        noise_power = torch.var(vec - reconstructed_points, unbiased=False)  # Power of the noise
        loss = -10 * torch.log10(signal_power / noise_power)
    else:
        loss = criterion(reconstructed_points, vec)
    optimizer.zero_grad()
    loss.backward()
    print(f"loss:{loss.item()}")
    optimizer.step()
    return hex_mat
