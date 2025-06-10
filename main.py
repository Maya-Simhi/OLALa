import gc
import sys
from statistics import mean
import time
import torch
import copy
import gc


from matplotlib import pyplot as plt
from sympy.logic.boolalg import Boolean

from DNNQuantize import PyTorchMLP
from alphModel import AlphhMLP
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federatedUtils
from torchinfo import summary
import numpy as np
from vit_pytorch import ViT
import torchvision.models as models1
import torch.nn as nn
import dataGetUtils
from torch.utils.data import ConcatDataset


def runningFL(args, user_matrix_old = None, global_matrix_old = None, train_matrix_alph_per_user_old = None):
    # data
    train_data, test_loader = utils.data(args)
    input, output = utils.get_dim(train_data, len(test_loader.dataset), args)
    global_test_data_loader = federatedUtils.load_mnist_data_test_global(args.train_batch_size, args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input, output, args.data)
    elif args.model == 'cnn3':
        global_model = models.CNN3Layer()
    else:
        global_model = models.Linear(input, output)
    textio.cprint(str(summary(global_model)))
    global_model.to(args.device)
    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # gives every user it's own data and optimizer
    local_models = federatedUtils.federated_setup(global_model, args) 
    #creating validation data for global user where combining all validation data
    # Initialize an empty list to store datasets
    combined_datasets = []
    for user_idx in range(args.num_users):
        user = local_models[user_idx]
        # Access the validation dataset for the current user
        validation_dataset = user['validationDataSet']  # Assuming 'user' stores datasets per user
        
        # Append the dataset to the combined_datasets list
        combined_datasets.append(validation_dataset)

    # Concatenate all the validation datasets from users
    combined_dataset = ConcatDataset(combined_datasets)

    # Create a new DataLoader for the combined dataset
    global_val_data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=args.train_batch_size, shuffle=True)

    # initilzation
    anotherModel = copy.deepcopy(global_model)
    train_matrix_per_user = [] #model of matrix training for each user
    train_matrix_alph_per_user = [] # model of training overloading value for each user
    train_optimizer_per_user = [] #optimizer of model for each user
    train_creation_per_user = []# creating of model for each user
    val_acc_list = []
    val_val_acc_list = []
    val_acc_per_user = [[] for _ in range(args.num_users)] #list of accuracy per user on test
    val_real_acc_per_user = [[] for _ in range(args.num_users)] #list of accuracy on validate per user
    snr_per_user = [[] for _ in range(args.num_users)]# list of snr per user
    utils.printStr(f'new test: code global iterations: {args.global_epochs}, local itertaions: {args.local_iterations}, training matrix {args.train_matrix_bool}, avraging: {args.average_bool} training global: {args.train_global_bool}, type of avrage : {args.average_type} num of users: {args.num_users}, joined digits? :{args.joined_digits}, model {args.model} data: {args.data}')
    for user_idx in range(args.num_users):
        train_matrix_per_user.append(PyTorchMLP(output_dim=args.lattice_dim))
        train_matrix_alph_per_user.append(AlphhMLP())
        train_creation_per_user.append(torch.nn.MSELoss())
        train_optimizer_per_user.append(torch.optim.Adam(train_matrix_per_user[user_idx].parameters(), lr=args.matrix_lr))
    globalModelAlpha = AlphhMLP()
    print("startin eterations")
    for global_epoch in range(0, args.global_epochs):
        print(f"globla epoch {global_epoch}")
        gc.collect()  # Free CPU memory
        torch.cuda.empty_cache()  # Free GPU memory
        if args.average_bool: #copy weights of global model to all users
            for user_idx in range(len(local_models)):
                combined_grads, shapes = dataGetUtils.getWeights(global_model)
                local_models[user_idx]['model'] = dataGetUtils.restoreWeights(shapes, combined_grads, local_models[user_idx]['model'])
        combined_grads, shapes = dataGetUtils.getWeights(global_model)
        state_dict_global = combined_grads
        user_matrix = []
        print("starting to train users")
        for user_idx in range(args.num_users):
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                print(f"train user: {user_idx}")
                result = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion,
                                                    args, state_dict_global, train_matrix_per_user[user_idx], train_matrix_alph_per_user[user_idx],
                                               user,train_optimizer_per_user[user_idx], train_creation_per_user[user_idx],
                                               args.train_matrix_bool, anotherModel)
                train_loss = result['loss']
                global_matrix = result['matrix']
                matrix_each = result['matrix_each']
                if args.lr_scheduler:
                    user['scheduler'].step(train_loss)
            user_matrix.append(matrix_each)

        utils.printStr(f'user matrix: {user_matrix}')
        if user_matrix_old is not None:
            user_matrix = user_matrix_old
        if global_matrix_old is not None:
            global_matrix = global_matrix_old
        if train_matrix_alph_per_user_old is not None:
            train_matrix_alph_per_user = train_matrix_alph_per_user_old
        res = federatedUtils.aggregate_models(local_models, global_model, global_matrix, user_matrix, args,
                                                snr_per_user, train_matrix_alph_per_user )  # FeaAvg
        global_model = res['global_model']
        if args.average_bool:
            val_acc, val_loss = dataGetUtils.test(global_test_data_loader, global_model, test_creterion, args.device)
            val_acc_2, _ = dataGetUtils.test(global_val_data_loader, global_model, test_creterion, args.device)
            for user_idx in range(args.num_users):
                user = local_models[user_idx]
                val_acc_1 = dataGetUtils.test(user['testData'],  user['model'], test_creterion, args.device)
                val_acc_per_user[user_idx].append(val_acc_1)
                val_acc_1 = dataGetUtils.test(user['validationData'],  user['model'], test_creterion, args.device)
                val_real_acc_per_user[user_idx].append(val_acc_1)

            val_acc_list.append(val_acc)
            val_val_acc_list.append(val_acc_2)
    if(args.train_with_alpha ):
        mystr = args.model_load_name
        if(args.train_matrix_bool):
            for user_idx, acc_list in enumerate(val_acc_per_user):
                mystr = args.model_load_name
                mystr = mystr + '_' + str(user_idx)
                torch.save(train_matrix_alph_per_user[user_idx].state_dict(), mystr)  
        else:
            torch.save(globalModelAlpha.state_dict(), mystr)  

    utils.printStr(f"val test acc global {val_acc_list}")
    utils.printStr(f"val validation acc global {val_val_acc_list}")
    if len(val_acc_list) >= 5:
        # Compute the average of the last 5 values
        avg_last_5 = sum(val_acc_list[-5:]) / len(val_acc_list[-5:])

        # Compute the average of the last 3 values
        avg_last_3 = sum(val_acc_list[-3:]) / len(val_acc_list[-3:])

        # Print results
        print(f"Average of last 5 test: {avg_last_5:.4f}")
        print(f"Average of last 3 test: {avg_last_3:.4f}")
        print(f"Average of last 5 val: {sum(val_val_acc_list[-5:]) / len(val_val_acc_list[-5:]):.4f}")
        print(f"Average of last 3 val: {sum(val_val_acc_list[-3:]) / len(val_val_acc_list[-3:]):.4f}")
    else:
        print("val acc list smaller then 5")

    for user_idx, acc_list in enumerate(val_acc_per_user):
        print(f"Test validate User {user_idx}: {acc_list}")
    for user_idx, acc_list in enumerate(val_real_acc_per_user):
        print(f"Real validate User {user_idx}: {acc_list}")
    for user_idx, acc_list in enumerate(snr_per_user):
        print(f"User snr {user_idx}: {acc_list}")

    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
    return user_matrix, global_matrix, train_matrix_alph_per_user
if __name__ == '__main__':
    # Check if GPU is available
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))


    if(args.train_global_bool): # we are training global model. steps: 1. train one user with all the data. 2. get matrix train 5 users with matrix. 3. train 10 users with matrix
        args.average_bool = False
        args.num_users = 1
        args.train_matrix_bool = True
        args.average_during_each = False
        user_matrix, global_matrix, train_matrix_alph_per_user = runningFL(args)
        args.num_users = 5
        args.train_global_bool = False
        args.average_bool = True
        args.train_matrix_bool = False
        args.average_type = "global"
        train_matrix_alph_per_user = runningFL(args, None, user_matrix[0], None)
        # args.num_users = 10
        # args.train_global_bool = False
        # args.average_bool = True
        # args.train_matrix_bool = False
        # args.average_type = "global"
        # train_matrix_alph_per_user = runningFL(args, None, user_matrix[0], None)
    elif(args.train_matrix_bool and args.average_bool and (not args.average_during_each)):  # we are training each model (user dependent). traing users without avraging and matrix model. after get matrix and train users with avraging
        args.average_bool = False
        user_matrix, global_matrix, train_matrix_alph_per_user = runningFL(args)
        args.train_matrix_bool = False
        args.average_bool = True
        args.average_type = "each"
        print(user_matrix)
        runningFL(args, user_matrix, global_matrix, train_matrix_alph_per_user)
    elif args.average_during_each: #online learning - just training matrix and avraging at the same time
        args.average_bool = True
        args.average_type = "each"
        runningFL(args)

    else:
        runningFL(args)



    