import sys
sys.path.append('../')

from advertorch import attacks
import argparse
import torch
import os
import numpy as np
import logging
import random
from tqdm import tqdm
from torchvision import datasets, transforms, models
from ensemble_model import EnsembleModel
from itertools import combinations
from sklearn import metrics
from model import fashionmnist_model
from train_engine import test_loss_ensemble, test_single
from attack_engine import attack_loss_ensemble, attack_single
from attack_engine import prepare_attack_example_loss_ensemble, prepare_attack_example_output_ensemble, prepare_attack_example_single
from util import init_log, set_seed, new_folder

parser = argparse.ArgumentParser(description='attack ensemble')
parser.add_argument('--data', default='../data/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--data_name', default='fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image_size', default=32, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='N', help='batch size (default: 32)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--n_epochs', default=40, type=int, metavar='N',
                    help='')
parser.add_argument('--save_model_path', default='../checkpoint/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--save_adv_path', default='../save_adv/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--prepare_attack_path', default='../prepare_attack_data/fashion_mnist', type=str,
                    help='path to dataset (e.g. data/')

parser.add_argument('--path_suffix', default='resnet20', type=str,
                    help='resnet20, resnet26, resnet32, resnetmix')
parser.add_argument('--model_num', default=4, type=int,
                    help='model name list')
parser.add_argument('--attack_method', default='FGSM', type=str,
                    help='FGSM, MI-FGSM, PGD, CW')
parser.add_argument('--train_type', default='loss_ensemble', type=str,
                    help='single, loss_ensemble, output_sensemble')
parser.add_argument('--loss_type', default='norm_cos3', type=str,
                    help='ce, project_loss, GPMR, norm_cos, norm_cos1')

parser.add_argument('--dynamic_type', default='normal', type=str,
                    help='normal, dynamic')
parser.add_argument('--para_config', default={}, type=dict,
                    help='normal, dynamic')
parser.add_argument('--alpha', default=0, type=float,
                    help='')
parser.add_argument('--para_flag', default=0, type=int,
                    help='')
parser.add_argument('--model_type', default='checkpoint', type=str,
                    help='checkpoint, model_best')

tqdm.monitor_interval = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path_suffix_list = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
loss_type_list = ['norm_cos1']
# loss_type_list = ['norm_cos1']

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

def load_model_list(args):
    model_list, model_name = get_model_list(args.model_num)

    if args.train_type == 'single':
        if args.loss_type == 'ce':
            for i, name in enumerate(model_name):
                logging.info(os.path.join(args.save_model_path, model_name[i]+'_single_checkpoint.pth.tar'))
                dict = torch.load(os.path.join(args.save_model_path, model_name[i]+'_single_model_best.pth.tar'))
                model_list[i].load_state_dict(dict)
        elif args.loss_type == 'regular_loss':
            for i, name in enumerate(model_name):
                logging.info(os.path.join(args.save_model_path, model_name[i]+ '_' + args.loss_type + '_single_checkpoint.pth.tar'))
                dict = torch.load(os.path.join(args.save_model_path, model_name[i]+ '_' + args.loss_type + '_single_model_best.pth.tar'))
                model_list[i].load_state_dict(dict)
        return model_list

    elif args.train_type == 'loss_ensemble':
        for i, name in enumerate(model_name):
            t = os.path.join(args.save_model_path, model_name[i]+'_'+args.dynamic_type+'_'+args.loss_type+'_loss_ensemble_checkpoint.pth.tar')

            if args.para_flag == True:
                dict = torch.load(os.path.join(args.save_model_path, str(args.para_config) + '_' + model_name[i] +'_' + args.dynamic_type+'_'+args.loss_type+'_loss_ensemble_' + args.model_type + '.pth.tar'))
            else:
                dict = torch.load(os.path.join(args.save_model_path, model_name[i] + '_' + args.dynamic_type+'_'+args.loss_type + '_loss_ensemble_' + args.model_type + '.pth.tar'))


            model_list[i].load_state_dict(dict)
        return model_list
    elif args.train_type == 'output_ensemble':
        file_name_suffix = '_'.join(model_name)
        model = torch.load(os.path.join(args.save_model_path, file_name_suffix+'_output_ensemble_checkpoint.pth.tar'))
        return model

def main_loss_ensemble(data_loader_test):
    model_list = load_model_list(args)              #加载重新训练的resnet
    _, model_name = get_model_list(args.model_num)

    if use_gpu:
        for i, model in enumerate(model_list):
            model_list[i] = model.cuda()
    cost = torch.nn.CrossEntropyLoss()

    clean_acc = prepare_attack_example_loss_ensemble(model_list, model_name, cost, data_loader_test, args)

    if args.para_flag == True:
        inputs = np.load(os.path.join(args.prepare_attack_path,     #../prepare_attack_data/fashion_mnist
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'x.npy'))
        targets = np.load(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'y.npy'))
    else:
        inputs = np.load(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                                      args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'x.npy'))
        targets = np.load(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'y.npy'))

    data = torch.utils.data.TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)

    optimzer_list = []
    for model in model_list:
        optimzer = torch.optim.Adam(model.parameters())
        optimzer_list.append(optimzer)

    state = {'model_list': model_list,
             'model_name': model_name,
             'optimzer_list': optimzer_list,
             'cost': cost,
             'args': args,
             'data_loader': data_loader,
             'clean_acc': clean_acc,
             }
    attack_loss_ensemble(state)

def main_single(data_loader_test):
    _, model_name = get_model_list(args.model_num)
    model_list = load_model_list(args)

    if use_gpu:
        for i, model in enumerate(model_list):
            model_list[i] = model.cuda()
    cost = torch.nn.CrossEntropyLoss()

    inputs_list = []
    targets_list = []
    data_loader_list = []
    for model, name in zip(model_list, model_name):
        prepare_attack_example_single(model, name, cost, data_loader_test, args)

        inputs = np.load(os.path.join(args.prepare_attack_path,     #../prepare_attack_data/fashion_mnist
                             'single',
                             args.loss_type + '_' + name + '_' + 'x.npy'))
        targets = np.load(os.path.join(args.prepare_attack_path,
                             'single',
                             args.loss_type + '_' + name + '_' + 'y.npy'))

        data = torch.utils.data.TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
        data_loader_list.append(data_loader)

    optimzer_list = []
    for model in model_list:
        optimzer = torch.optim.Adam(model.parameters())
        optimzer_list.append(optimzer)

    state = {'model_list': model_list,
             'model_name': model_name,
             'optimzer_list': optimzer_list,
             'cost': cost,
             'args': args,
             'data_loader_list': data_loader_list,
             }
    attack_single(state)

if __name__ == '__main__':
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    for path_suffix in path_suffix_list:
        args.path_suffix = path_suffix

        args.save_model_path = '../checkpoint/fashion_mnist'
        args.save_adv_path = '../save_adv/fashion_mnist'
        args.prepare_attack_path = '../prepare_attack_data/fashion_mnist'

        args.para_config = {'alpha': args.alpha}  # 0../checkpoint/fashion_mnist   resnetmix
        log_save_path = os.path.join(args.save_model_path, args.path_suffix + '_log', '')
        new_folder(log_save_path)

        if args.para_flag == 1:
            args.para_flag = True
        else:
            args.para_flag = False

        # if args.para_flag == True:
        #     init_log(os.path.join(log_save_path,
        #                           str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))
        # else:
        #     init_log(os.path.join(log_save_path,
        #                           args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))

        args.save_model_path = os.path.join(args.save_model_path, args.path_suffix, '')
        new_folder(args.save_model_path)
        args.save_adv_path = os.path.join(args.save_adv_path, args.path_suffix, '')
        new_folder(args.save_adv_path)
        new_folder(os.path.join(args.save_adv_path, args.train_type, ''))
        args.prepare_attack_path = os.path.join(args.prepare_attack_path, args.path_suffix, '')
        new_folder(args.prepare_attack_path)
        new_folder(os.path.join(args.prepare_attack_path, args.train_type, ''))


    # for loss_type in ['ce', 'project_loss', 'GPMR', 'norm_cos']:

        for loss_type in loss_type_list:
            args.loss_type = loss_type

            if args.para_flag == True:
                init_log(os.path.join(log_save_path,
                                      str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))
            else:
                init_log(os.path.join(log_save_path,
                                      args.dynamic_type + '_' + args.loss_type + args.train_type + '.log'))

            #for attack_alg in ['BIM']:
            for attack_alg in ['FGSM', 'MI-FGSM', 'PGD', 'BIM']:
                args.attack_method = attack_alg
                print(args)

                use_gpu = torch.cuda.is_available()

                # set seed
                set_seed()

                transform_test = transforms.Compose(
                    [transforms.ToTensor(),
                     ])

                data_test = datasets.FashionMNIST(root="../data", transform=transform_test, train=False)
                data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                               batch_size=args.batch_size,
                                                               shuffle=False,
                                                               num_workers=args.workers)

                if args.train_type == 'single':
                    main_single(data_loader_test)
                elif args.train_type == 'loss_ensemble':
                    main_loss_ensemble(data_loader_test)
