import argparse

import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")

    # data arguments
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='cnn2',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users participating in the federated learning")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=100,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--global_epochs', type=int, default=40, 
                        help="number of global epochs")
    parser.add_argument('--joined_digits', type=bool, default=True,
                        help="if we want the user to have a joined digits or not")
    parser.add_argument('--joined_digits_num', type=int, default=3,
                        help="how many digits do you want to join? only works on 10 users")
    parser.add_argument('--should_split_data_evenly', type=bool, default=False,
                    help="should we split data evenly")
    parser.add_argument('--should_use_diff_in_quantizer', type=bool, default=False,
                    help="should we get the wighets or the diff of the wighets?") 

    # Matrix learned configurations
    parser.add_argument('--train_matrix_bool', type=bool, default=True,
                        help="if we want to train a matrix")
    parser.add_argument('--average_bool', type=bool, default=True,
                        help="should we do averaging")
    parser.add_argument('--average_type', type=str, default='hexagon', 
                        choices=['none', 'hexagon', 'global', 'each', 'D2', 'A2'],
                        help="what avreging do you want")
                        # The A2 and D2 came from https://www.math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/A2.html
                        # The lattice came from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=508838&tag=1
    parser.add_argument('--train_global_bool', type=bool, default=False, 
                        help="do we want to train global matrix?")
    parser.add_argument('--train_with_alpha', type=bool, default=False,
                        help="if we want to train a alph model")
    parser.add_argument('--model_load_name', type=str, default='alpha',
                        help="what name do you want the model to have")
    parser.add_argument('--average_during_each', type=bool, default=True,
                        help="should we do averaging during learning each?")
    parser.add_argument('--num_codewords', type=int, default=64, 
                        help="number of code words in quantizer")
    parser.add_argument('--matrix_lr', type=float, default=1e-6, 
                        help="learning rate of the matrix shoud be around 1e-5")
    parser.add_argument('--modulo_of_matrix', type=int, default=10,
                        help="how many iterations should we do until we train the matrix")
    parser.add_argument('--loss_by', type=str, default="mse",
                        choices=['mse', 'accuracy', 'snr'],
                        help="what loss we want in matrix learning")
    parser.add_argument('--num_overloading', type=int, default=10,
                       help="the overloading we want if we want( good overloading is between 0.3 to 10). -1 is for logic with variance")

    


    # privacy arguments
    parser.add_argument('--privacy', action='store_true', default=True,
                        help="whether to preserve privacy")
    parser.add_argument('--privacy_noise', type=str, default='jopeq_vector',
                        choices=['laplace', 't', 'jopeq_scalar', 'jopeq_vector'],
                        help="types of PPNs to choose from")
    parser.add_argument('--epsilon', type=float, default=4,
                        help="privacy budget (epsilon)")
    parser.add_argument('--sigma_squared', type=float, default=0.2,
                        help="scale for t-dist Sigma (identity matrix)")
    parser.add_argument('--nu', type=float, default=4,
                        help="degrees of freedom for t-dist")
    # (epsilon, sigma_squared, nu): (0.84,3,3), (1,2,3), (2,1,12), (2,0.5,4), (3,1,32), (3,0.5,14), (3,0.2,4), (4,0.85,50), (4,0.1,3), (4,0.2,4), (4,0.5,29)

    # quantization arguments
    parser.add_argument('--quantization', action='store_true', default=True,
                        help="whether to perform quantization")
    parser.add_argument('--lattice_dim', type=int, default=2,
                        choices=[1, 2],
                        help="perform scalar (lattice_dim=1) or lattice (lattice_dim=2) quantization ")
    parser.add_argument('--R', type=int, default=18,
                        help="compression rate (number of bits per sample)")
    parser.add_argument('--gamma', type=float,
                        #default = 0.5,
                        default=set_gamma(parser.parse_args()),
                        help="quantizer dynamic range")
    parser.add_argument('--vec_normalization', type=bool, default=set_vec_normalization(parser.parse_args()),
                        help="whether to perform vectorized normalization, otherwise perform scalar normalization")

    # learning arguments
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate  for MNIST 0.1  and CIFAR 0.003 ")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_false',
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=1234,
                        help="manual seed for reproducibility")
    parser.add_argument('--should_use_seed', type=bool, default=True,
                        help="should we use seed?")

    args = parser.parse_args()
    return args


def set_gamma(args):
    eta = 1.5
    gamma = 1
    if args.privacy:
        if args.lattice_dim == 1:
            gamma += 2 * ((2 / args.epsilon) ** 2)
        else:
            gamma += args.sigma_squared * (args.nu / (args.nu - 2))
    return eta * np.sqrt(gamma)



def set_vec_normalization(args):
    vec_normalization = False
    if args.quantization:
        if args.lattice_dim == 2:
            vec_normalization = True
    if args.privacy:
        if args.privacy_noise == 't' or args.privacy_noise == 'jopeq_vector':
            vec_normalization = True
    return vec_normalization

