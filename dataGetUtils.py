import os
from statistics import mean
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import quantizer

import logging


def test(test_loader, model, criterion, device, shouldKeppGrads = False):
    if shouldKeppGrads:
        shouldPrint = False
        model.eval()  # Set model to eval mode, but keep gradients
        correct = 0
        test_loss = torch.tensor(0.0, device=device, requires_grad=True)  
        shouldPrintOnce = shouldPrint
    

        for data, label in test_loader:
            data, label = data.to(device), label.to(device)  # Move to device
            output = model(data)

            loss = criterion(output, label)  # Compute loss
            test_loss = test_loss + loss  

            pred = output.argmax(dim=1, keepdim=True)  # Get index of max log-probability

            if shouldPrintOnce:
                print(f"Predicted: {pred.view(-1).tolist()}, Label: {label.view(-1).tolist()}")
                shouldPrintOnce = False


            correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        if shouldPrint:
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Test Loss: {test_loss.item():.4f}")  # Print only at the end

        return accuracy, test_loss/ len(test_loader.dataset)

    else:
        model.eval()
        correct = 0
        test_loss = 0
        shouldPrint = True

        with torch.no_grad():  # No need to track gradients
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)  # Move to device
                output = model(data)

                loss = criterion(output, label)  # Compute loss
                test_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)  # Get index of max log-probability

                if shouldPrint:
                    print(f"Predicted: {pred.view(-1).tolist()}, Label: {label.view(-1).tolist()}")
                    shouldPrint = False

                correct += pred.eq(label.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        test_loss /= len(test_loader)  # Average loss per batch

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        return accuracy, test_loss



# Function to retrieve and flatten the weights of the model
def getWeights(model):
    modified_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only consider trainable parameters
            modified_weights[name] = param.data.clone()
    # Combine all weights into one flattened tensor
    flattened_weights = []
    shapes = {}
    for name, weight in modified_weights.items():
        shapes[name] = weight.shape  # Save the original shape
        flattened_weights.append(weight.view(-1))  # Flatten the weight tensor

    # Concatenate all flattened weights into a single tensor
    combined_weights = torch.cat(flattened_weights)
    return combined_weights, shapes


# Function to restore weights back to the model while keeping it differentiable
def restoreWeights(shapes, combined_weights, model):
    
    restored_weights = {}
    offset = 0
    for name, shape in shapes.items():
        num_elements = torch.prod(torch.tensor(shape)).item()
        
        # Instead of detaching, ensure we preserve gradients
        restored_weights[name] = combined_weights[offset:offset + num_elements].view(shape)
        offset += num_elements

    # Assign restored weights to the model **without** detaching gradients
    for name, param in model.named_parameters():
        if param.requires_grad and name in restored_weights:
            param.data = restored_weights[name] 

    return model
