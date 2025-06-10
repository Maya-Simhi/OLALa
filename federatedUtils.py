import numpy as np
import torch
import torch.optim as optim
import copy
import math

import quantizer
import torchvision.transforms as transforms
import torchvision
import utils
import sys
import dataGetUtils


def federated_setup(global_model, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    shouldGlobal = args.train_global_bool
    local_models = {}
    for user_idx in range(args.num_users):
        train_dataset, val_loader, val_dataset = load_mnist_data(args.train_batch_size, user_idx,shouldGlobal, args)
        user = {'data': train_dataset, 'testData': load_mnist_data_test(args.train_batch_size, user_idx, shouldGlobal, args),
                'validationData': val_loader,  'validationDataSet': val_dataset,
            'model': copy.deepcopy(global_model)}
        user['opt'] = (optim.SGD(user['model'].parameters(), lr=args.lr, momentum=args.momentum) 
                                if args.optimizer == 'sgd' 
                                else optim.Adam(user['model'].parameters(), lr=args.lr))
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def SNRCalc(reconstructed_points, vec):
    signal_power = torch.var(vec, unbiased=False)  # Power of the signal
    noise_power = torch.var(vec - reconstructed_points, unbiased=False)  # Power of the noise
    return 10 * torch.log10(signal_power / noise_power)

def getFilter(args, shouldGlobal, indexName):
    if shouldGlobal:
        train_filter = lambda x: (x[1] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    else:
        if args.joined_digits:
            num = 1
            if args.num_users == 5:
                num = 2
            #     if indexName == 4:
            #         train_filter = lambda x: x[1] in [8,9,0]
            #     else:
            #         train_filter = lambda x: x[1] in [indexName*2, indexName*2 + 1, indexName*2 +2]
            if args.num_users == 10 or args.num_users == 5:
                if args.joined_digits_num == 2:
                    train_filter = lambda x: x[1] in [indexName*num, (indexName*num + 1)%10]
                elif args.joined_digits_num == 3:
                    train_filter = lambda x: x[1] in [indexName*num, (indexName*num + 1)%10,  (indexName*num + 2)%10]                
                elif args.joined_digits_num == 4:
                    train_filter = lambda x: x[1] in [indexName*num, (indexName*num + 1)%10,  (indexName*num + 2)%10,  (indexName*num + 3)%10]
                elif args.joined_digits_num == 1:
                    train_filter = lambda x: x[1] in [indexName]
                elif args.joined_digits_num == 5:
                    train_filter = lambda x: x[1] in [indexName*num, (indexName*num + 1)%10,  (indexName*num + 2)%10,  (indexName*num + 3)%10, (indexName*num + 4)%10]
            if args.num_users == 2:
                if indexName == 1:
                    train_filter = lambda x: x[1] in [5,6,7,8,9,0]
                else:
                    train_filter = lambda x: x[1] in [0,1,2,3,4,5]
        else:
            if args.num_users == 10:
                train_filter = lambda x: x[1] in [indexName]
            if args.num_users == 5:
                train_filter = lambda x: x[1] in [indexName*2, indexName*2 + 1]
    return train_filter

def load_mnist_data(batch_size, indexName, shouldGlobal, args):

    transform = transforms.Compose([transforms.ToTensor()])
    if args.data == 'mnist':
        # Load the full MNIST datasets
        train_dataset = torchvision.datasets.MNIST(root='/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=True, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(32), #ensure image size is correct.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.CIFAR10('/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=True, download=False,
                                     transform=transform
                                    )
    if args.should_split_data_evenly:
        subset_size = len(train_dataset) // 5  

        # Generate random indices for the subset
        indices = torch.randperm(len(train_dataset))[:subset_size]

        # Create subset using indices (much faster than random_split)
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        return train_subset

     # if we don't want to split evenly we split based on digits
    train_filter = getFilter(args, shouldGlobal, indexName)
  

    # Filter the datasets using list comprehension (alternative to Subset)
    train_dataset_filtered = [x for x in train_dataset if train_filter(x)]

     # Split into train and validation (80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset_filtered))
    val_size = len(train_dataset_filtered) - train_size

    # Random split into training and validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_filtered, [train_size, val_size])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    return train_dataset, val_loader, val_dataset

def load_mnist_data_test(batch_size, indexName, shouldGlobal,args):
    transform = transforms.Compose([transforms.ToTensor()])
    if args.data == 'mnist':
        # Load the full MNIST datasets
        train_dataset = torchvision.datasets.MNIST(root='/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=False, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(32), #ensure image size is correct.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.CIFAR10('/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=False, download=False,
                                    transform=transform
                                    )

    if args.should_split_data_evenly:
        print("spliting")
        subset_size = len(train_dataset) // 5  

        # Generate random indices for the subset
        indices = torch.randperm(len(train_dataset))[:subset_size]

        # Create subset using indices (much faster than random_split)
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        return  torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # if we don't want to split evenly we split based on digits
    train_filter = getFilter(args, shouldGlobal, indexName)
    # Filter the datasets using list comprehension (alternative to Subset)
    train_dataset_filtered = [x for x in train_dataset if train_filter(x)]

    # Create DataLoaders for the filtered datasets
    train_loader = torch.utils.data.DataLoader(train_dataset_filtered, batch_size=batch_size, shuffle=True)

    return train_loader

def load_mnist_data_test_global(batch_size, args):
    transform = transforms.Compose([transforms.ToTensor()])

    if args.data == 'mnist':
        # Load the full MNIST datasets
        train_dataset = torchvision.datasets.MNIST(root='/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=False, download=False, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(32), #ensure image size is correct.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.CIFAR10('/home/mayasimh/DeepLatticeUVEQ-main/JoPEQ-main/data', train=False, download=False,
                                     transform=transform
                                    )

    
    if args.should_split_data_evenly:
        print("spliting")
        subset_size = len(train_dataset) // 5  

        # Generate random indices for the subset
        indices = torch.randperm(len(train_dataset))[:subset_size]

        # Create subset using indices (much faster than random_split)
        train_subset = torch.utils.data.Subset(train_dataset, indices)
        return torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)

     # if we don't want to split evenly we split based on digits here it's global so we take all
    train_filter = lambda x: x[1] in [0, 1, 2, 3, 4, 5,6,7,8,9] #todo change id numbers grow

    # Filter the datasets using list comprehension (alternative to Subset)
    train_dataset_filtered = [x for x in train_dataset if train_filter(x)]

    # Create DataLoaders for the filtered datasets
    train_loader = torch.utils.data.DataLoader(train_dataset_filtered, batch_size=batch_size, shuffle=True)

    return train_loader


def getGrads(modelUser):
    modified_grads = {}
    for name, param in modelUser.named_parameters():
        if param.grad is not None:
            modified_grads[name] = param.grad.clone()
    # Combine all gradients into one tensor
    flattened_grads = []
    shapes = {}
    for name, grad in modified_grads.items():
        shapes[name] = grad.shape  # Save the original shape
        flattened_grads.append(grad.view(-1))  # Flatten the gradient tensor

    # Concatenate all flattened gradients into one tensor
    combined_grads = []
    if(len(flattened_grads) != 0):
        combined_grads = torch.cat(flattened_grads)
    return combined_grads,  shapes
def restoreGrads(shapes, combined_grads, model):
    # Create a dictionary to hold the reshaped gradients
    restored_grads = {}
    offset = 0
    for name, shape in shapes.items():
        # Compute the size of the current gradient tensor
        num_elements = torch.prod(torch.tensor(shape)).item()
        # Slice the corresponding part of the combined tensor and reshape
        restored_grads[name] = combined_grads[offset:offset + num_elements].view(shape)
        offset += num_elements
    # Copy the restored gradients back to the model
    for name, param in model.named_parameters():
        if name in restored_grads:
            param.grad = restored_grads[name]
    return model

def aggregate_models(local_models, global_model,global_matrix, user_matrix, args, snr_per_user = [], train_matrix_alph_per_user = None):  # FeaAvg
    #doing FL
    if args.average_bool:
        gen_mat_lattice = np.array([[1, 1 / 2], [0, np.sqrt(3) / 2]])
        lattice_gen = torch.from_numpy(gen_mat_lattice).to(torch.float32).to(args.device)
        sumWighets = 0
        firstRun = True
        combined_grads_glob, shapes = dataGetUtils.getWeights(global_model)
        for user_idx in range(0, len(local_models)):
            modelUser = local_models[user_idx]['model']
            combined_grads, shapes = dataGetUtils.getWeights(modelUser)
            if args.should_use_diff_in_quantizer:
                combined_grads = combined_grads - combined_grads_glob
            torch.set_printoptions(threshold=10_000)
            #lattice
            if args.average_type != "none":
                if args.average_type == "global":
                    print(global_matrix)
                    matrix = global_matrix
                elif args.average_type == "hexagon":
                    matrix = lattice_gen
                elif args.average_type == "each":
                    print(user_matrix)
                    matrix = user_matrix[user_idx]
                elif args.average_type == "D2":
                    D2_lattice = np.array([[2, 0], [1, -1]])
                    matrix = torch.from_numpy(D2_lattice).to(torch.float32).to(args.device)
                elif args.average_type == "A2":
                    D2_lattice = np.array([[np.sqrt(2), 0], [-0.707106781187, 1.22474487139]])
                    matrix = torch.from_numpy(D2_lattice).to(torch.float32).to(args.device)
                else:
                    print("got incoorect type")
                    sys.exit()
                
                mechanism = quantizer.LatticeQuantization(args,
                                                        matrix, True)
                if args.train_with_alpha:
                    train_matrix_alph_per_user[user_idx].eval()
                    gettingAlph = train_matrix_alph_per_user[user_idx](combined_grads).to(args.device)
                    combined_grads, vec = mechanism(input =combined_grads,gettingAlph = gettingAlph, shouldPrint =False, shouldReturnBack = True)
                else:
                   # print(f"user: {user_idx}: matrix: {matrix}, before: {combined_grads}")
                    combined_grads, vec = mechanism(combined_grads, False, True)
                    #print(f"user: {user_idx} after: {combined_grads}")
                snr_per_user[user_idx].append(SNRCalc(combined_grads, vec))
            if(firstRun):
                print("in here no grads")
                firstRun = False
                sumWighets =  combined_grads
            else:
                sumWighets = sumWighets + combined_grads

        if args.should_use_diff_in_quantizer:
            sumWighets = sumWighets /(len(local_models))
            sumWighets = sumWighets + combined_grads_glob
        else:
            sumWighets = sumWighets/(len(local_models))
        global_model = dataGetUtils.restoreWeights(shapes, sumWighets ,  global_model)
        return { 'global_model': global_model}

    return { 'global_model': global_model }