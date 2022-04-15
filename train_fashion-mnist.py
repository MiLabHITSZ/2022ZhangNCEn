import sys
sys.path.append('../')

import shutil
import argparse
import torch
import os
import numpy as np
import logging
from tqdm import tqdm
from torchvision import datasets, transforms, models
from ensemble_model import EnsembleModel
from itertools import combinations
from sklearn import metrics
from train_engine import train_loss_ensemble, train_single
from model import fashionmnist_model
import torch.optim as optim
import random
import torch.distributions as tdist
from util import init_log, set_seed, new_folder

parser = argparse.ArgumentParser(description='train ensemble')
parser.add_argument('--data', default='../data/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--data_name', default='fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image_size', default=28, type=int,
                    metavar='N', help='image size (default: 28)')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='N', help='batch size (default: 64)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save_model_path', default='../checkpoint/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--begin_epoch', default=0, type=int, metavar='N',
                    help='')
parser.add_argument('--n_epochs', default=40, type=int, metavar='N',
                    help='')
parser.add_argument('--path_suffix', default='resnet20', type=str,
                    help='resnet20, resnet26, resnet32, resnetmix')
parser.add_argument('--model_num', default=5, type=int,
                    help='model number')
parser.add_argument('--train_type', default='loss_ensemble', type=str,
                    help='single, loss_ensemble')
parser.add_argument('--loss_type', default='norm_cos', type=str,
                    help='ce, project_loss, GPMR, norm_cos, norm_cos1')
parser.add_argument('--dynamic_type', default='normal', type=str,
                    help='normal')
parser.add_argument('--para_config', default={}, type=dict,
                    help='normal, dynamic')
parser.add_argument('--alpha', default=0, type=float,
                    help='')
parser.add_argument('--para_flag', default=0, type=int,
                    help='')
parser.add_argument('--para_proj', default=1, type=float,
                    help='para before project_loss')
parser.add_argument('--para_norm', default=0, type=float,
                    help='para before norm_loss')
parser.add_argument('--para_cos', default=0.02, type=float,
                    help='para before cos_loss')

tqdm.monitor_interval = 0

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model_list(model_num):
    if args.path_suffix == 'resnet20':
        model_list = [fashionmnist_model.resnet20() for i in range(model_num)]
        model_name = ['resnet20-'+str(i) for i in range(model_num)]
    if args.path_suffix == 'resnet26':
        model_list = [fashionmnist_model.resnet26() for i in range(model_num)]
        model_name = ['resnet26-'+str(i) for i in range(model_num)]
    if args.path_suffix == 'resnet32':
        model_list = [fashionmnist_model.resnet32() for i in range(model_num)]
        model_name = ['resnet32-'+str(i) for i in range(model_num)]
    if args.path_suffix == 'resnetmix':
        if args.model_num == 3:
            model_list = [fashionmnist_model.resnet20(),
                          fashionmnist_model.resnet26(),
                          fashionmnist_model.resnet32()]
        if args.model_num == 4:
            model_list = [fashionmnist_model.resnet20(),
                          fashionmnist_model.resnet20(),
                          fashionmnist_model.resnet26(),
                          fashionmnist_model.resnet32()]
        if args.model_num == 5:
            model_list = [fashionmnist_model.resnet20(),
                          fashionmnist_model.resnet20(),
                          fashionmnist_model.resnet26(),
                          fashionmnist_model.resnet26(),
                          fashionmnist_model.resnet32()]
        model_name = ['resnetmix-' + str(i) for i in range(model_num)]

    return model_list, model_name

def main_loss_ensemble(data_loader_train, data_loader_test):
    model_list, model_name = get_model_list(args.model_num)

    if use_gpu:
        for model in model_list:
            model.cuda()
    cost = torch.nn.CrossEntropyLoss()

    optimzer_list = []
    scheduler_list = []
    for model in model_list:
        optimzer = optim.Adam(model.parameters())  # 学习率默认是1e-3
        scheduler = optim.lr_scheduler.StepLR(optimzer, step_size=15, gamma=0.1)
        optimzer_list.append(optimzer)
        scheduler_list.append(scheduler)

    if 'dynamic' in args.dynamic_type:
        for model in model_list:
            optimzer = optim.Adam(model.parameters())           #学习率默认是1e-3
            scheduler = optim.lr_scheduler.StepLR(optimzer, step_size=15, gamma=0.1)
            optimzer_list.append(optimzer)
            scheduler_list.append(scheduler)

    state = {'model_list': model_list,
             'model_name': model_name,
             'optimzer_list': optimzer_list,
            'scheduler_list': scheduler_list,
             'cost': cost,
             'args': args,
             'data_loader_train': data_loader_train,
             'data_loader_test': data_loader_test,
             'use_gpu': use_gpu
             }
    train_loss_ensemble(state, args.para_proj)

def main_single(data_loader_train, data_loader_test):
    model_list, model_name = get_model_list(args.model_num)

    if use_gpu:
        for i, model in enumerate(model_list):
            model_list[i] = model.cuda()
    cost = torch.nn.CrossEntropyLoss()

    optimzer_list = []
    scheduler_list = []
    for model in model_list:
        optimzer = optim.Adam(model.parameters())  # 学习率默认是1e-3
        scheduler = optim.lr_scheduler.StepLR(optimzer, step_size=15, gamma=0.1)
        optimzer_list.append(optimzer)
        scheduler_list.append(scheduler)

    state = {'model_list': model_list,
             'model_name': model_name,
             'optimzer_list': optimzer_list,
             'scheduler_list': scheduler_list,
             'cost': cost,
             'args': args,
             'data_loader_train': data_loader_train,
             'data_loader_test': data_loader_test,
             'use_gpu': use_gpu
             }
    train_single(state)

if __name__ == '__main__':
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    args.para_config = {'alpha': args.alpha}
    print(args)
    use_gpu = torch.cuda.is_available()

    args.save_model_path = os.path.join(args.save_model_path, args.path_suffix, '')
    new_folder(args.save_model_path)

    if args.para_flag == 1:
        args.para_flag = True
    else:
        args.para_flag = False
    if args.para_flag == True:
        init_log(os.path.join(args.save_model_path,
                              args.path_suffix + '_log',
                              str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))
    else:
        init_log(os.path.join(args.save_model_path,
                              args.path_suffix + '_log',
                              args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))
    # set seed
    set_seed()

    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, 4),
         transforms.ToTensor()
         ])
    transform_train_noise = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: torch.clamp(x + torch.empty(x.shape).normal_(mean=0.0, std=0.3 / 2), 0, 1))
         ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         ])

    data_train = datasets.FashionMNIST(root="../data",
                                       transform=transform_train,
                                       train=True,
                                       download=True)
    data_train_noise = datasets.FashionMNIST(root="../data",
                                             transform=transform_train_noise,
                                             train=True,
                                             download=True)
    data_test = datasets.FashionMNIST(root="../data",
                                      transform=transform_test,
                                      train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset((data_train, data_train_noise)),
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.workers)

    if args.train_type == 'single':
        main_single(data_loader_train, data_loader_test)
    elif args.train_type == 'loss_ensemble':
        main_loss_ensemble(data_loader_train, data_loader_test)