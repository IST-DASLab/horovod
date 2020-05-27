import argparse
import yaml
import os

import torch


def add_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch NUQSGD')

    # options overwritting yaml options
    parser.add_argument('--path_opt', default='default.yaml',
                        type=str, help='path to a yaml options file')
    parser.add_argument('--data', default=argparse.SUPPRESS,
                        type=str, help='path to data')
    parser.add_argument('--logger_name', default='runs/runX')
    parser.add_argument('--dataset', default='mnist', help='mnist|cifar10')

    # options that can be changed from default
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int, default=argparse.SUPPRESS, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=argparse.SUPPRESS,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=argparse.SUPPRESS,
                        metavar='N',
                        help='how many batches to wait before logging training'
                             ' status')
    parser.add_argument('--tblog_interval',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--optim', default=argparse.SUPPRESS, help='sgd|dmom')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default=argparse.SUPPRESS,
                        help='model architecture: (default: resnet32)')
    parser.add_argument('-j', '--workers', default=argparse.SUPPRESS,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--weight_decay', '--wd', default=argparse.SUPPRESS,
                        type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--train_accuracy', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_profiler', action='store_true')
    parser.add_argument('--lr_decay_epoch',
                        default=argparse.SUPPRESS)
    parser.add_argument('--log_keys', default='')
    parser.add_argument('--exp_lr',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--nodropout',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--data_aug',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--noresume', action='store_true',
                        help='do not resume from checkpoint')
    parser.add_argument('--pretrained',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--num_class',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--lr_decay_rate',
                        default=argparse.SUPPRESS, type=float)
    parser.add_argument('--nesterov',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--run_dir', default='runs/runX')
    parser.add_argument('--ckpt_name', default='checkpoint.pth.tar')
    parser.add_argument('--g_estim', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--epoch_iters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_log_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_estim_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--gvar_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_optim',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--g_optim_start',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_osnap_iter',
                        default='100,1000,10000', type=str)
    parser.add_argument('--g_bsnap_iter',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--g_epoch',
                        default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--niters',
                        default=argparse.SUPPRESS, type=int)
    parser.add_argument('--no_batch_norm',
                        default=argparse.SUPPRESS, type=bool)
    # NUQ
    parser.add_argument('--nuq_method', default='q', help='q|nuq|qinf')
    parser.add_argument('--nuq_bits', default=4, type=int)
    parser.add_argument('--nuq_bucket_size', default=1024, type=int)
    parser.add_argument('--nuq_ngpu', default=1, type=int)
    parser.add_argument('--nuq_mul', default=0.5, type=float)
    parser.add_argument('--nuq_amq_lr',
                        default=0.7, type=float)
    parser.add_argument('--nuq_amq_epochs',
                        default=50, type=int)
    parser.add_argument('--untrain_steps', default=0, type=int)
    parser.add_argument('--untrain_lr', default=0.001, type=float)
    parser.add_argument('--untrain_std', default=0.001, type=float)
    parser.add_argument('--nuq_sym', default=False, action='store_true')
    parser.add_argument('--nuq_inv', default=False, action='store_true')
    parser.add_argument('--nuq_parallel', default='no', help='no|gpu1|ngpu')
    parser.add_argument('--dist_num', default=20, type=int)
    parser.add_argument('--chkpt_iter', default=20, type=int)
    parser.add_argument('--nuq_number_of_samples',
                        default=argparse.SUPPRESS,
                        type=int,
                        help='NUQ Number of Samples')
    parser.add_argument('--nuq_ig_sm_bkts',
                        action='store_true',
                        help='NUQ Ignore Small Buckets')
    parser.add_argument('--nuq_truncated_interval',
                        default=argparse.SUPPRESS,
                        type=float,
                        help='NUQ Truncated Interval')
    parser.add_argument('--nuq_cd_epochs', default=argparse.SUPPRESS,
                        help='NUQ Adaptive CD Epochs', type=int)
    parser.add_argument('--nuq_layer', action='store_true',
                        help='NUQ Enable Network Wide Quantization')
    args = parser.parse_args()
    return args


def opt_to_nuq_kwargs(opt):
    return {
        'ngpu': opt.nuq_ngpu, 'bits': opt.nuq_bits,
        'bucket_size': opt.nuq_bucket_size, 'method': opt.nuq_method,
        'multiplier': opt.nuq_mul, 'cd_epochs': opt.nuq_cd_epochs,
        'number_of_samples': opt.nuq_number_of_samples,
        'path': opt.logger_name, 'symmetric': opt.nuq_sym,
        'interval': opt.nuq_truncated_interval, 'amq_epochs': opt.nuq_amq_epochs,
        'learning_rate': opt.nuq_learning_rate, 'amq_lr': opt.nuq_amq_lr,
        'ig_sm_bkts': opt.nuq_ig_sm_bkts, 'inv': opt.nuq_inv
    }
