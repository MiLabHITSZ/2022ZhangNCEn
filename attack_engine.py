import sys
sys.path.append('../')

from advertorch import attacks
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
from train_engine import test_loss_ensemble, test_single
import xlwt
import xlrd
from xlutils.copy import copy

num_classes = 10
clip_min = 0.0
clip_max = 1.0
eps = 0.1
use_gpu = True

def get_diversity_loss(x_list, loss_type):
    if loss_type == 'gal_loss':
        #caculate diversity loss (get predcit loss gradient for x)
        x_combinations = list(combinations(x_list, 2))
        loss = 0
        for combine in x_combinations:
            cosine_similarity = torch.cosine_similarity(combine[0], combine[1])
            cosine_similarity_exp = torch.exp(cosine_similarity)
            if (cosine_similarity_exp > loss):
                loss = cosine_similarity_exp

        loss = torch.log(loss).cpu().numpy()
        return loss

    elif loss_type == 'sign_loss':
        loss = torch.exp(torch.sum(torch.sign(x_list), 0))
        return loss

def measure(y, output, predict):
    #f1 = metrics.f1_score(y, predict, average='micro')
    acc = metrics.accuracy_score(y, predict)
    #fpr, tpr, _ = metrics.roc_curve(y, output)
    #auc = metrics.auc(fpr, tpr)
    return acc

def get_attack_adversary(args, model, attack_method):
    adversary_list = []
    if attack_method == 'FGSM':
        if args.data_name == 'mnist':
            eps_list = [0.1, 0.3]
        elif args.data_name == 'fashion_mnist':
            eps_list = [0.1, 0.3]
        elif args.data_name == 'cifar10':
            eps_list = [0.03, 0.09]
        for eps in eps_list:
            adversary = attacks.GradientSignAttack(model,
                                                   loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                                   eps=eps,
                                                   clip_min=clip_min,
                                                   clip_max=clip_max,
                                                   targeted=False)
            adversary_list.append(adversary)
        return adversary_list, eps_list

    if attack_method == 'MI-FGSM':
        if args.data_name == 'mnist':
            eps_list = [0.1, 0.3]
        elif args.data_name == 'fashion_mnist':
            eps_list = [0.1, 0.3]
        elif args.data_name == 'cifar10':
            eps_list = [0.03, 0.09]
        for eps in eps_list:
            adversary = attacks.MomentumIterativeAttack(model,
                                                        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                                        eps=eps,
                                                        clip_min=clip_min,
                                                        clip_max=clip_max,
                                                        targeted=False
                                                        )
            adversary_list.append(adversary)
        return adversary_list, eps_list

    elif attack_method == 'PGD':
        if args.data_name == 'mnist':
            eps_list = [0.1, 0.15]
        elif args.data_name == 'fashion_mnist':
            eps_list = [0.1, 0.15]
        elif args.data_name == 'cifar10':
            eps_list = [0.01, 0.02]
        for eps in eps_list:
            adversary = attacks.PGDAttack(model,
                                          loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                          eps=eps,
                                          nb_iter=40,
                                          eps_iter=eps,
                                          rand_init=True,
                                          clip_min=clip_min,
                                          clip_max=clip_max,
                                          ord = np.inf,
                                          l1_sparsity = None,
                                          targeted = False)
            adversary_list.append(adversary)
        return adversary_list, eps_list

    elif attack_method == 'CW':
        if args.data_name == 'mnist':
            const_list = [0.1, 10]
        elif args.data_name == 'fashion_mnist':
            const_list = [0.1, 10]
        elif args.data_name == 'cifar10':
            const_list = [0.001, 0.1]
        for const in const_list:
            adversary = attacks.CarliniWagnerL2Attack(model,
                                                      num_classes=num_classes,
                                                      confidence=0,
                                                      targeted=False,
                                                      learning_rate=0.01,
                                                      binary_search_steps=9,
                                                      max_iterations=1000,
                                                      abort_early=True,
                                                      initial_const=const,
                                                      clip_min=clip_min,
                                                      clip_max=clip_max,
                                                      loss_fn=None)
            adversary_list.append(adversary)
        return adversary_list, const_list

    if attack_method == 'BIM':
        if args.data_name == 'mnist':
            eps_list = [0.1, 0.15]
        elif args.data_name == 'fashion_mnist':
            eps_list = [0.1, 0.15]
        elif args.data_name == 'cifar10':
            eps_list = [0.01, 0.02]
        for eps in eps_list:
            adversary = attacks.LinfBasicIterativeAttack(model,
                                                   loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                                   eps=eps,
                                                   nb_iter=10,
                                                   eps_iter=eps,
                                                   clip_min=clip_min,
                                                   clip_max=clip_max,
                                                   targeted=False)
            adversary_list.append(adversary)
        return adversary_list, eps_list
    if attack_method == 'JSMA':
        if args.data_name == 'mnist':
            eps_list = [0.3, 0.6]
            theta = 0.2
        elif args.data_name == 'fashion_mnist':
            eps_list = [0.3, 0.6]
            theta = 0.2
        elif args.data_name == 'cifar10':
            eps_list = [0.05, 0.1]
            theta = 0.1
        for eps in eps_list:
            adversary = attacks.JacobianSaliencyMapAttack(model,
                                                          num_classes=10,
                                                          clip_min=clip_min,
                                                          clip_max=clip_max,
                                                          loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                                                          theta=theta,
                                                          gamma=eps,
                                                          comply_cleverhans=False)
            adversary_list.append(adversary)
        return adversary_list, eps_list

def write_csv(clean_acc, acc, args, i, j):
    rb = xlrd.open_workbook('result.xls')
    wb = copy(rb)
    if args.data_name == 'fashion_mnist':
        if args.attack_method == 'FGSM':
            row = 2 + i
        elif args.attack_method == 'MI-FGSM':
            row = 4 + i
        elif args.attack_method == 'PGD':
            row = 6 + i
        elif args.attack_method == 'BIM':
            row = 8 + i
        # elif args.attack_method == 'CW':
        #     row = 10 + i
        # elif args.attack_method == 'JSMA':
        #     row = 12 + i

    elif args.data_name == 'cifar10':
        if args.attack_method == 'FGSM':
            row = 12 + i
        elif args.attack_method == 'MI-FGSM':
            row = 14 + i
        elif args.attack_method == 'PGD':
            row = 16 + i
        elif args.attack_method == 'BIM':
            row = 18 + i
        # elif args.attack_method == 'CW':
        #     row = 22 + i
        # elif args.attack_method == 'JSMA':
        #     row = 26 + i

    if args.loss_type == 'norm_cos':
        col = 2
    elif args.loss_type == 'norm_cos1':
        col = 3
    elif args.loss_type == 'norm_cos2':
        col = 4
    elif args.loss_type == 'norm_cos3':
        col = 5
    elif args.loss_type == 'norm_cos4':
        col = 6
    elif args.loss_type == 'norm_cos5':
        col = 7
    elif args.loss_type == 'norm_cos6':
        col = 8
    elif args.loss_type == 'norm_cos7':
        col = 9
    elif args.loss_type == 'norm_cos8':
        col = 10
    elif args.loss_type == 'norm_cos9':
        col = 11
    elif args.loss_type == 'norm_cos10':
        col = 12
    elif args.loss_type == 'norm_cos11':
        col = 13


    if args.path_suffix == 'resnet20':
        if args.data_name == 'fashion_mnist':
            wb.get_sheet(0).write(1, col, '{:.4f}\n'.format(clean_acc))
        elif args.data_name == 'cifar10':
            wb.get_sheet(0).write(11, col, '{:.4f}\n'.format(clean_acc))
    elif args.path_suffix == 'resnet26':
        if args.data_name == 'fashion_mnist':
            wb.get_sheet(0).write(21, col, '{:.4f}\n'.format(clean_acc))
        elif args.data_name == 'cifar10':
            wb.get_sheet(0).write(31, col, '{:.4f}\n'.format(clean_acc))
        row = row + 20
    elif args.path_suffix == 'resnet32':
        if args.data_name == 'fashion_mnist':
            wb.get_sheet(0).write(41, col, '{:.4f}\n'.format(clean_acc))
        elif args.data_name == 'cifar10':
            wb.get_sheet(0).write(51, col, '{:.4f}\n'.format(clean_acc))
        row = row + 40
    elif args.path_suffix == 'resnetmix':
        if args.data_name == 'fashion_mnist':
            wb.get_sheet(0).write(61, col, '{:.4f}\n'.format(clean_acc))
        elif args.data_name == 'cifar10':
            wb.get_sheet(0).write(71, col, '{:.4f}\n'.format(clean_acc))
        row = row + 60

    wb.get_sheet(0).write(row, col, '{:.4f}\n'.format(acc))
    wb.save('result.xls')

def attack_loss_ensemble(state):
    model_list = state['model_list']
    model_name = state['model_name']
    cost = state['cost']
    args = state['args']
    data_loader_attack = state['data_loader']
    model_len = len(model_list)
    clean_acc = state['clean_acc']

    ensemble_model = EnsembleModel(model_list, model_name, ensemble_type='predict_mean')
    ensemble_model.eval()
    adversary_list, para_list = get_attack_adversary(args, ensemble_model, args.attack_method)
    for i, (adversary, para) in enumerate(zip(adversary_list, para_list)):
        logging.info(args.attack_method + ':' + str(para))
        adv_example = []
        preds = []
        targets = []
        outputs = []
        data_loader_attack = tqdm(data_loader_attack, desc='Attack', ncols=60)
        for data in data_loader_attack:
            # logging.info("train ing")
            x, y = data
            if use_gpu:
                x, y = x.cuda(), y.cuda()

            adv_untargeted = adversary.perturb(x, y)

            adv_example.extend(adv_untargeted.detach().cpu().numpy())
            output = ensemble_model(adv_untargeted)
            _, pred = torch.max(output, 1)

            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            outputs.extend((output.detach().cpu().numpy()))

        acc = measure(targets, outputs, preds)
        logging.info(acc)
        write_csv(clean_acc, acc, args, i, 0)

        adv_example = np.asarray(adv_example)
        save_path = os.path.join(args.save_adv_path,            #../save_adv/fashion_mnist/resnetmix/loss_ensemble
                                 'loss_ensemble',
                                  args.attack_method + '_' + str(para) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'x.npy')
        np.save(save_path, adv_example)

def attack_single(state):
    model_list = state['model_list']
    model_name = state['model_name']
    optimzer_list = state['optimzer_list']
    cost = state['cost']

    args = state['args']
    data_loader_list = state['data_loader_list']
    model_len = len(model_list)

    for i, model in enumerate(model_list):
        model.eval()
        logging.info("model:{}".format(model_name[i]))
        adversary_list, para_list = get_attack_adversary(args, model, args.attack_method)

        for j, (adversary, para) in enumerate(zip(adversary_list, para_list)):
            logging.info(args.attack_method + ':' + str(para))
            data_loader = tqdm(data_loader_list[i], desc='Attack', ncols=60)
            adv_example = []
            preds = []
            targets = []
            outputs = []
            for data in data_loader:
                # logging.info("train ing")
                x, y = data
                if use_gpu:
                    x, y = x.cuda(), y.cuda()

                adv_untargeted = adversary.perturb(x, y)

                adv_example.extend(adv_untargeted.detach().cpu().numpy())
                output = model(adv_untargeted)
                _, pred = torch.max(output, 1)

                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
                outputs.extend((output.detach().cpu().numpy()))

            acc = measure(targets, outputs, preds)
            logging.info(acc)
            # write_csv(0, acc, args, j, i)

            adv_example = np.asarray(adv_example)
            save_path = os.path.join(args.save_adv_path,
                                     'single',
                                     args.attack_method + '_' + str(para) + '_' + args.loss_type + '_' + model_name[i] + '_'  + 'x.npy')
            np.save(save_path, adv_example)

def prepare_attack_example_output_ensemble(ensemble_model, model_name, cost, data_loader_test):
    test_loss = 0
    outputs = []
    targets = []
    preds = []
    inputs = []
    data_loader_test = tqdm(data_loader_test, desc='Test')
    ensemble_model.eval()

    with torch.no_grad():
        for data in data_loader_test:
            # logging.info("train ing")
            x_test, y_test = data
            if use_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()
            output = ensemble_model(data)
            _, pred = torch.max(output.cpu().detach().numpy(), 1)
            test_loss += cost(output, y_test).item()

            inputs.extend(x_test.cpu().numpy())
            outputs.extend(output.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            targets.extend(y_test.cpu().numpy())

    image_save_names = [i for i in range(len(inputs))]
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    acc = measure(targets, outputs, preds)
    logging.info(metrics)

    true_idx = (preds == targets)
    image_save_names = image_save_names[true_idx]
    inputs = inputs[true_idx]
    outputs = outputs[true_idx]
    preds = preds[true_idx]
    targets = targets[true_idx]

    # np.save(os.path.join(args.prepare_attack_path,
    #                      'output_ensemble',
    #                      model_name + '_' + 'x.npy'), inputs)
    # np.save(os.path.join(args.prepare_attack_path,
    #                      'output_ensemble',
    #                      model_name + '_' + 'output.npy'), outputs)
    # np.save(os.path.join(args.prepare_attack_path,
    #                      'output_ensemble',
    #                      model_name + '_' + 'predict.npy'), preds)
    # np.save(os.path.join(args.prepare_attack_path,
    #                      'output_ensemble',
    #                      model_name + '_' + 'y.npy'), targets)

def prepare_attack_example_loss_ensemble(model_list, model_name, cost, data_loader_test, args):
    test_loss = 0
    outputs = []
    targets = []
    preds = []
    inputs = []
    data_loader_test = tqdm(data_loader_test, desc='Test')
    ensemble_model = EnsembleModel(model_list, model_name, ensemble_type='predict_mean')
    ensemble_model.eval()           #测试模式，保证神经网络的参数值不会发生变化

    with torch.no_grad():           #tensor不会自动求导
        for data in data_loader_test:
            # logging.info("train ing")
            x_test, y_test = data
            if use_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()
            output = ensemble_model(x_test)
            _, pred = torch.max(output, 1)
            test_loss += cost(output, y_test).item()

            inputs.extend(x_test.cpu().numpy())
            outputs.extend(output.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            targets.extend(y_test.cpu().numpy())

    image_save_names = [i for i in range(len(inputs))]
    inputs = np.asarray(inputs)         #tensor
    outputs = np.asarray(outputs)       #概率
    preds = np.asarray(preds)           #标签
    targets = np.asarray(targets)       #标签

    acc = measure(targets, outputs, preds)
    logging.info(acc)

    true_idx = (preds == targets)
    image_save_names = np.asarray([i for i in range(len(inputs))])
    #输出预测正确的样本的各项内容
    inputs = inputs[true_idx]
    outputs = outputs[true_idx]
    preds = preds[true_idx]
    targets = targets[true_idx]

    if args.para_flag == True:
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'x.npy'), inputs)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_'+ 'output.npy'), outputs)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'predict.npy'), preds)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                             str(args.para_config) + '_' + args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'y.npy'), targets)
    else:
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                                                           args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'x.npy'), inputs)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                                                           args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'output.npy'), outputs)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                                                           args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'predict.npy'), preds)
        np.save(os.path.join(args.prepare_attack_path,
                             'loss_ensemble',
                                                           args.dynamic_type + '_' + args.loss_type + '_' + str(model_name) + '_' + 'y.npy'), targets)

    return acc

def prepare_attack_example_single(model, model_name, cost, data_loader_test, args):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    preds = []
    inputs = []
    data_loader_test = tqdm(data_loader_test, desc='Test')

    with torch.no_grad():
        for data in data_loader_test:
            # logging.info("train ing")
            x_test, y_test = data
            if use_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()
            output = model(x_test)
            inputs.extend(x_test.cpu().numpy())
            _, pred = torch.max(output, 1)
            test_loss += cost(output, y_test).item()

            outputs.extend(output.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            targets.extend(y_test.cpu().numpy())

    image_save_names = np.asarray([i for i in range(len(inputs))])
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    acc = measure(targets, outputs, preds)
    logging.info(acc)
    true_idx = (preds == targets)
    image_save_names = image_save_names[true_idx]
    inputs = inputs[true_idx]
    outputs = outputs[true_idx]
    preds = preds[true_idx]
    targets = targets[true_idx]

    np.save(os.path.join(args.prepare_attack_path,
                         'single',
                         args.loss_type + '_' + model_name + '_' + 'x.npy'), inputs)
    np.save(os.path.join(args.prepare_attack_path,
                         'single',
                         args.loss_type + model_name + '_' + 'output.npy'), outputs)
    np.save(os.path.join(args.prepare_attack_path,
                         'single',
                         args.loss_type + '_' + model_name + '_' + 'predict.npy'), preds)
    np.save(os.path.join(args.prepare_attack_path,
                         'single',
                         args.loss_type + '_' + model_name + '_' + 'y.npy'), targets)
