import os
from statistics import mean
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from DNNQuantize import train_model
import federatedUtils
import quantizer
import dataGetUtils

import logging
import random



# Configure logging
logging.basicConfig(
    filename="FL3_10_users.log",  # Log file name
    level=logging.INFO,            # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    datefmt="%Y-%m-%d %H:%M:%S"    # Date format
)

def printStr(myStr):
    print(myStr)
    # logging.info(myStr)


def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def get_dim(data, amount, args):
    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output


def train_one_epoch(train_data, model,
                    optimizer, creterion,
                     args, stateDictGlobal, quantizierModleEach, quantizierModleAlphaEach,user, optimizerlattice,
                    creationLattice, shouldTrainMatrix = False,anotherModel = None):
    device = args.device
    iterations = args.local_iterations
    model.train()
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')
    losses = []
    counter = 0
    #shuffel each time
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True  # Shuffle data at the beginning of each epoch
    )

    if iterations is not None:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        counter = counter + 1
        #traning quantizer
        if shouldTrainMatrix:
            if counter%args.modulo_of_matrix ==0:
                combined_grads, _ = dataGetUtils.getWeights(model)
                if args.should_use_diff_in_quantizer:
                    combined_grads = combined_grads - stateDictGlobal
                matrix_each = train_model(quantizierModleEach, quantizierModleAlphaEach, args,creationLattice, 1e-6, combined_grads, optimizerlattice, train_loader, model, anotherModel )
        optimizer.step()


        losses.append(loss.item())

        if iterations is not None:
            local_iteration += 1
            if local_iteration == iterations:
                break
    ret = {"loss": mean(losses),
        "matrix":torch.tensor([[ 0.9793, -0.9523],
        [-0.9842, -0.9561]], requires_grad=True),
        "matrix_each":torch.tensor([[ 0.9793, -0.9523],
         [-0.9842, -0.9561]])}
    if shouldTrainMatrix:
        ret = {"loss": mean(losses),"matrix":torch.tensor([[ 0.9793, -0.9523],
         [-0.9842, -0.9561]]), "matrix_each": matrix_each}

    return ret


def initializations(args):
    if args.should_use_seed:
        #  reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.inf
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

