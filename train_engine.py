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
from torch.nn import functional
import matplotlib.pyplot as plt

use_gpu = True
# loss mask
mask = None


def text_save(filename, loss_type, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file_path = "/home/zhw/code/advexmp/Projection/result/" + filename + '-' + loss_type + ".txt"
    # file_path = "C:\\Users\\ER2\\OneDrive\\桌面\\code\\advexmp\\Projection\\result\\" + filename + '-' + loss_type + ".txt"
    r_file = open(file_path, 'w+')  # 会对原有内容清空并有读写权限

    for i in range(len(data)):
        # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        # s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        s = str(data[i])
        s = s.replace("'", '').replace(',', '') + '\n'

        r_file.write(s)
    r_file.close()
    print("保存文件成功")


# 得到两个已经归一化的梯度的和  即对角线梯度
def get_tensor_angel_divide(x_grads, y_grads):
    # angle_divide = data_normal(x_grads) + data_normal(y_grads)
    angle_divide = x_grads + y_grads
    return angle_divide


def measure(y, output, predict):  # y为真实类别，predict为预测类别
    acc = metrics.accuracy_score(y, predict)
    metric = {
        'acc': acc,
    }
    logging.info(str(metric))
    return acc


def save_checkpoint(model, is_best, best_score, save_model_path, suffix, filename='checkpoint.pth.tar'):
    logging.info('best score:{} '.format(best_score))
    filename_ = suffix + filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    logging.info('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)

    if is_best:
        filename_best = suffix + 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)
        logging.info('save model {filename}'.format(filename=filename_best))
        filename_best = os.path.join(save_model_path,
                                     suffix + 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        logging.info('save model {filename}'.format(filename=filename_best))
        shutil.copyfile(filename, filename_best)


def test_loss_ensemble(model_list, optimzer_list, model_name, cost, data_loader_test):
    pred_loss = 0
    gal_loss = 0
    model_num = len(model_list)
    outputs = []
    preds = []
    targets = []

    ensemble_model = EnsembleModel(model_list, model_name, ensemble_type='predict_mean')
    ensemble_model.eval()
    data_loader_test = tqdm(data_loader_test, desc='Test', ncols=60)

    with torch.no_grad():
        for data in data_loader_test:
            x_test, y_test = data
            if use_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()

            # get every model output
            output = ensemble_model(x_test)
            _, pred = torch.max(output, 1)  # _代表概率  pred预测类别

            # save prediction result
            outputs.extend(output.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            targets.extend(y_test.cpu().numpy())

    outputs = np.asarray(outputs)
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    acc = measure(targets, outputs, preds)
    return acc


def test_single(model, cost, data_loader_test):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    preds = []
    data_loader_test = tqdm(data_loader_test, desc='Test', ncols=60)
    num_class = 10

    with torch.no_grad():
        for data in data_loader_test:
            # print("train ing")
            x_test, y_test = data
            if use_gpu:
                x_test, y_test = x_test.cuda(), y_test.cuda()
            output = model(x_test)
            _, pred = torch.max(output, 1)
            test_loss += cost(output, y_test).item()

            pred = functional.one_hot(pred, num_classes=num_class)
            y_test = functional.one_hot(y_test, num_classes=num_class)

            outputs.extend(output.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            targets.extend(y_test.cpu().numpy())

    outputs = np.asarray(outputs)
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    acc = measure(targets, outputs, preds)
    return acc


def get_project_loss(x_grads, model_list, diversity_loss_type, para_a):
    if diversity_loss_type == 'project_lossnew':  # 按照余弦相似性的值从大到小排序   取距离基准向量最近的一个
        proj_loss = 0
        new_grads = []
        disance = []
        for i in range(len(x_grads)):
            new_grads.append(x_grads[i])
            a = x_grads[i].view(x_grads[i].size(0), -1)
            if i > 0:
                for j in range(i):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)
            if i < len(x_grads):
                for j in range(i + 1, len(x_grads)):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)

            dis_cos = torch.stack(disance)
            aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序
            sort = sort_indices[0]  # 1*64维的索引
            for index in range(args.batch_size):
                if sort[index] < i:
                    new_grads.append(x_grads[sort[index]])
                if sort[index] >= i:
                    new_grads.append(x_grads[sort[index] + 1])
                c = new_grads[0] + new_grads[index + 1]
                proj = torch.norm(c, p=2)
                proj_loss += proj
            new_grads.clear()
            disance.clear()

        project_loss = proj_loss.mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_5_5':
        project_loss = 0
        for i in range(len(x_grads)):
            project_loss += x_grads[i]

        project_loss = torch.norm(project_loss)
        project_loss = project_loss.mean()

        x_grads = torch.stack(x_grads)
        x_grads = x_grads.view(x_grads.size(0), x_grads.size(1), -1)
        loss = torch.norm(x_grads, dim=2)
        loss = torch.mul(loss, loss)
        loss = loss.mean(dim=1)
        loss = loss.sum()
        return 0.5 * loss + para_a * project_loss

    elif diversity_loss_type == 'project_loss_2_5':  # 归一化求欧氏距离  按欧氏距离进行排序
        proj_loss = 0
        new_grads = []
        disance = []
        new_grads.append(x_grads[0])
        for i in range(len(x_grads) - 1):
            a = x_grads[0].view(x_grads[0].shape[0], -1)
            b = x_grads[i + 1].view(x_grads[i + 1].shape[0], -1)  # 转换为2维

            a = functional.normalize(a)
            b = functional.normalize(b)  # 归一化

            disa = torch.dist(a, b, p=2)
            disance.append(disa)

        disance = torch.stack(disance)
        aaa, sort = torch.sort(disance, dim=0)
        for j in range(len(x_grads) - 1):
            new_grads.append(x_grads[sort[j] + 1])
        for k in range(len(x_grads) - 1):
            c = new_grads[k] + new_grads[k + 1]
            proj = torch.norm(c)
            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        # x_grads = torch.stack(x_grads)
        # x_grads = x_grads.view(x_grads.size(0), x_grads.size(1), -1)
        # loss = torch.norm(x_grads, dim=2)
        # loss = torch.mul(loss, loss)
        # loss = loss.mean(dim=1)
        # loss = loss.sum()
        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_2_5_1':  # 投影到基准向量（归一化）上排序
        proj_loss = 0
        new_grads = []
        disance = []
        new_grads.append(x_grads[0])
        a = x_grads[0] / torch.norm(x_grads[0], p=2)
        print('============')
        for i in range(len(x_grads) - 1):
            b = x_grads[i + 1] * torch.cosine_similarity(x_grads[i + 1], a)
            e = torch.cosine_similarity(x_grads[i + 1], a)
            print(e.shape)
            c = torch.norm(b, p=2)
            disance.append(c)

        disance = torch.stack(disance)
        aaa, sort = torch.sort(disance, dim=0, descending=True)
        for j in range(len(x_grads) - 1):
            new_grads.append(x_grads[sort[j] + 1])
        for k in range(len(x_grads) - 1):
            c = new_grads[k] + new_grads[k + 1]
            proj = torch.norm(c)
            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_2_5_new':  # 按欧氏距离进行排序  按序排列 1,2  2,3  3,4  4,5
        proj_loss = 0
        new_grads = []
        disance = []
        new_grads.append(x_grads[0])
        for i in range(len(x_grads) - 1):
            disance.append(torch.dist(x_grads[i + 1], x_grads[0], p=2))
        disance = torch.stack(disance)
        aaa, sort = torch.sort(disance, dim=0)
        for j in range(len(x_grads) - 1):
            new_grads.append(x_grads[sort[j] + 1])
        for k in range(len(x_grads) - 1):
            c = new_grads[k] + new_grads[k + 1]
            proj = torch.norm(c)
            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss


def get_project_loss_new(x_grads, diversity_loss_type, para_a):
    if diversity_loss_type == 'project_loss':  # 复现GAL的论文
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)

            proj_loss += torch.exp(cosine_similarity)

        proj_loss = torch.log(proj_loss).mean()

        return 0.5 * proj_loss

    elif diversity_loss_type == 'GPMR':
        div_loss = 0
        eq_loss = 0
        x_combinations = list(combinations(x_grads, 2))
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)
            div = cosine_similarity + (1 / (args.model_num - 1))
            div = torch.mul(div, div)

            div_loss += div
        div_loss = (2 / (args.model_num * (args.model_num - 1))) * div_loss
        div_loss = div_loss.mean()

        for i in range(args.model_num):
            eq1 = torch.norm(x_grads[i], p=2)
            eq2 = 0
            for j in range(args.model_num):
                eq2 += torch.norm(x_grads[j], p=2)
            eq2 = eq2 / args.model_num
            eq = eq1 - eq2
            eq = torch.mul(eq, eq)
            eq_loss += eq
        eq_loss = eq_loss / args.model_num
        eq_loss = eq_loss.mean()

        if args.data_name == 'cifar10':
            return 0.04 * div_loss + 10 * eq_loss

        elif args.data_name == 'fashion_mnist':
            return 0.1 * div_loss + 10 * eq_loss

        elif args.data_name == 'mnist':
            return 0.1 * div_loss + 10 * eq_loss

        else:
            print("unknown dataset!")


    elif diversity_loss_type == 'project_loss_sort':  # 按照余弦相似性的值从大到小   （1,2） （2,3） （3，4) (4,5)
        new_grads = []
        disance = []
        proj_loss = 0

        a = x_grads[0].view(x_grads[0].size(0), -1)
        for i in range(1, args.model_num):
            b = x_grads[i].view(x_grads[i].size(0), -1)
            cos_sim = torch.cosine_similarity(a, b, dim=1)
            disance.append(cos_sim)

        dis_cos = torch.stack(disance)
        aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序

        for index in range(len(sort_indices[0])):
            sort = sort_indices[:, index]
            grads = []
            grads.append(x_grads[0])
            for i in range(args.model_num - 1):
                grads.append(x_grads[sort[i] + 1])
            for j in range(args.model_num - 1):
                c = grads[j] + grads[j + 1]
                proj = torch.norm(c, p=2)
                proj_loss += proj

        project_loss = proj_loss.mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_cos':  # 按照余弦相似性的值从大到小排序   取距离基准向量最近的一个
        proj_loss = 0
        for i in range(len(x_grads)):
            new_grads = []
            disance = []
            # new_grads.append(x_grads[i])
            a = x_grads[i].view(x_grads[i].size(0), -1)
            if i > 0:
                for j in range(i):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)
            if i < len(x_grads):
                for j in range(i + 1, len(x_grads)):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)

            dis_cos = torch.stack(disance)
            aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序
            sort = sort_indices[0]  # 1*64维的索引
            # print(sort)
            for index in range(len(sort)):
                if sort[index] < i:
                    new_grads.append(x_grads[sort[index]][index])
                elif sort[index] >= i:
                    new_grads.append(x_grads[sort[index] + 1][index])
            new_grads = torch.stack(new_grads)
            # print(new_grads.shape)          #64*3*32*32
            # c = new_grads.view(new_grads.size(0), -1)
            proj = x_grads[i] + new_grads
            proj = torch.norm(proj, p=2)

            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_cos1':  # 按照余弦相似性的值从大到小排序   取距离基准向量最近的一个
        proj_loss = 0
        for i in range(len(x_grads)):
            new_grads = []
            disance = []
            # new_grads.append(x_grads[i])
            a = x_grads[i].view(x_grads[i].size(0), -1)
            if i > 0:
                for j in range(i):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)
            if i < len(x_grads):
                for j in range(i + 1, len(x_grads)):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)

            dis_cos = torch.stack(disance)
            aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序
            sort = sort_indices[0]  # 1*64维的索引
            # print(sort)
            for index in range(len(sort)):
                if sort[index] < i:
                    new_grads.append(x_grads[sort[index]][index])
                elif sort[index] >= i:
                    new_grads.append(x_grads[sort[index] + 1][index])
            new_grads = torch.stack(new_grads)
            # print(new_grads.shape)          #64*3*32*32
            # c = new_grads.view(new_grads.size(0), -1)
            proj = x_grads[i] + new_grads
            proj = torch.norm(proj, p=2)

            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss

    elif diversity_loss_type == 'VIP':
        proj_loss = 0
        for i in range(len(x_grads)):
            new_grads = []
            disance = []
            # new_grads.append(x_grads[i])
            a = x_grads[i].view(x_grads[i].size(0), -1)
            if i > 0:
                for j in range(i):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)
            if i < len(x_grads):
                for j in range(i + 1, len(x_grads)):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)

            dis_cos = torch.stack(disance)
            aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序
            sort = sort_indices[0]  # 1*64维的索引
            # print(sort)
            for index in range(len(sort)):
                if sort[index] < i:
                    new_grads.append(x_grads[sort[index]][index])
                elif sort[index] >= i:
                    new_grads.append(x_grads[sort[index] + 1][index])
            new_grads = torch.stack(new_grads)
            # print(new_grads.shape)          #64*3*32*32
            c = new_grads.view(new_grads.size(0), -1)
            proj = torch.mul(a, c)
            proj = torch.sum(proj, dim=1)

            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_cosnew':  # 按照余弦相似性进行排序，取距离基准向量最近的两个
        proj_loss = 0
        first_max_grads = []
        disance = []
        second_max_grads = []
        for i in range(len(x_grads)):
            a = x_grads[i].view(x_grads[i].size(0), -1)
            if i > 0:
                for j in range(i):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)
            elif i < len(x_grads):
                for j in range(i + 1, len(x_grads)):
                    b = x_grads[j].view(x_grads[j].size(0), -1)
                    cos_sim = torch.cosine_similarity(a, b, dim=1)
                    disance.append(cos_sim)

            dis_cos = torch.stack(disance)
            aaa, sort_indices = torch.sort(dis_cos, dim=0, descending=True)  # 从大到小排序
            first_sort = sort_indices[0]  # 1*64维的索引
            second_sort = sort_indices[1]
            for index in range(args.batch_size):
                if first_sort[index] < i:
                    first_max_grads.append(x_grads[first_sort[index]])
                elif first_sort[index] >= i:
                    first_max_grads.append(x_grads[first_sort[index] + 1])
                if second_sort[index] < i:
                    second_max_grads.append(x_grads[second_sort[index]])
                elif second_sort[index] >= i:
                    second_max_grads.append(x_grads[second_sort[index] + 1])
                c = x_grads[i] + first_max_grads[index] + second_max_grads[index]
                proj = torch.norm(c, p=2)
                proj_loss += proj
            first_max_grads.clear()
            second_max_grads.clear()
            disance.clear()

        project_loss = proj_loss

        return para_a * project_loss

    elif diversity_loss_type == 'project_loss_combine':
        # caculate diversity loss (get predcit loss gradient for x)
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            c = combine[0] + combine[1]
            proj = torch.norm(c, p=2)
            proj_loss += torch.exp(proj)

        project_loss = torch.log(proj_loss).mean()

        return para_a * project_loss

    elif diversity_loss_type == 'project_norm_sum':
        x_combinations = list(combinations(x_grads, 2))
        proj_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)

            c = torch.norm(a, dim=1, p=2)  # (64,1)维
            d = torch.norm((a * b), dim=1, p=2) / (torch.norm(a, dim=1, p=2) * (-1 / (args.model_num - 1)))

            c = torch.mul(c, c)
            d = torch.mul(d, d)
            proj = c + d
            proj_loss += proj

        proj_loss = proj_loss.mean()
        return para_a * proj_loss

    elif diversity_loss_type == 'gal_dis_regular_loss':
        x_combinations = list(combinations(x_grads, 2))  # 得到任意两个梯度的组合
        gal_loss = 0
        dis_loss = 0
        for combine in x_combinations:
            a = combine[0].view(combine[0].size(0), -1)
            b = combine[1].view(combine[1].size(0), -1)
            cosine_similarity = torch.cosine_similarity(a, b, dim=1)  # 求梯度a和b的余弦相似性
            dis = torch.norm(a, dim=1) - torch.norm(b, dim=1)  # 求a和b的范数差
            dis = torch.mul(dis, dis)

            gal_loss += torch.exp(cosine_similarity)
            dis_loss += torch.exp(dis)

        gal_loss = torch.log(gal_loss).mean()
        dis_loss = torch.log(dis_loss).mean()

        x_grads = torch.stack(x_grads)  # 正则损失  让所有模长都减小
        x_grads = x_grads.view(x_grads.size(0), x_grads.size(1), -1)
        print(x_grads.shape)
        loss = torch.norm(x_grads, dim=2)
        print(loss.shape)
        loss = torch.mul(loss, loss)
        loss = loss.mean(dim=1)
        print(loss.shape)
        loss = loss.sum()
        return 0.5 * loss + 0.5 * gal_loss + 0.5 * dis_loss


def get_project_loss_NC(x_grads, diversity_loss_type, para_norm, para_cos):
    if diversity_loss_type in ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5',
                               'norm_cos6', 'norm_cos7', 'norm_cos8', 'norm_cos9', 'norm_cos10', 'norm_cos11']:  # 除以g
        norm = torch.zeros(x_grads[0].size(0)).cuda()
        g = torch.zeros(x_grads[0].size(0), x_grads[0].size(1), x_grads[0].size(2), x_grads[0].size(3)).cuda()
        gn = torch.zeros(x_grads[0].size(0)).cuda()
        for i in range(args.model_num):
            g += x_grads[i] / args.model_num
            xi = x_grads[i].view(x_grads[i].size(0), -1)
            # print('111', torch.div(torch.norm(xi, dim=1), args.model_num))
            # print('222', gn)
            gn += torch.div(torch.norm(xi, dim=1), args.model_num)
            # print("=====", torch.norm(xi, dim=1).shape)   64

        for i in range(args.model_num):
            xi = x_grads[i].view(x_grads[i].size(0), -1)
            a = torch.norm(xi, dim=1) - gn
            a = torch.div(a, gn)    #******************
            b = torch.zeros(x_grads[0].size(0)).cuda()
            for j in range(args.model_num):
                if j != i:
                    xj = x_grads[j].view(x_grads[j].size(0), -1)
                    tmp = torch.norm(xj, dim=1) - gn
                    tmp = torch.div(tmp, gn)      #******************
                    b += tmp / args.model_num
            proj = torch.mul(a, b)
            norm += torch.exp(proj)
        norm_loss = torch.log(norm).mean()

        cos = torch.zeros(x_grads[0].size(0)).cuda()
        qn = g.view(g.size(0), -1)
        for i in range(args.model_num):
            fi = x_grads[i].view(x_grads[i].size(0), -1)
            cos1 = torch.cosine_similarity(fi, qn, dim=1)
            cos1 = torch.exp(cos1)
            cos2 = torch.zeros(x_grads[0].size(0)).cuda()
            for j in range(args.model_num):
                if j != i:
                    fj = x_grads[j].view(x_grads[j].size(0), -1)
                    tmp = torch.cosine_similarity(fj, qn, dim=1)
                    cos2 += torch.exp(tmp)
            cos += torch.mul(cos1, cos2)
        cos_loss = torch.log(cos).mean()
        return para_norm * norm_loss + para_cos * cos_loss

    # if diversity_loss_type in ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5',
    #                            'norm_cos6', 'norm_cos7', 'norm_cos8', 'norm_cos9', 'norm_cos10', 'norm_cos11']:
    #
    #     g = torch.zeros(x_grads[0].size(0), x_grads[0].size(1), x_grads[0].size(2), x_grads[0].size(3)).cuda()
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #     qn = g.view(g.size(0), -1)
    #     #qn还可以用ensemble求
    #
    #     nc_loss = torch.zeros(x_grads[0].size(0)).cuda()
    #     #与每一个分量负相关
    #     for i in range(args.model_num):
    #         a = x_grads[i].view(x_grads[i].size(0), -1) - qn    #########
    #         b_norm = torch.zeros(x_grads[0].size(0)).cuda()
    #         for j in range(args.model_num):
    #             if j != i:
    #                 b = x_grads[j].view(x_grads[i].size(0), -1) - qn    #########
    #                 # print('a.shape', a.shape)
    #                 # print('b.shape', b.shape)
    #                 dot = torch.mul(a, b)
    #                 # print('dot.shape', dot.shape)
    #                 b_norm += torch.sum(dot, dim=1)
    #                 # b_norm += dot_sum
    #         nc_loss += torch.exp(b_norm)
    #     norm_loss = torch.log(nc_loss).mean()
    #     return para_cos * norm_loss

        #与整体负相关
        # for i in range(args.model_num):
        #     a = x_grads[i].view(x_grads[i].size(0), -1) + qn
        #     b_norm = 0
        #     bn = 0
        #     for j in range(args.model_num):
        #         if j != i:
        #             b = x_grads[j].view(x_grads[i].size(0), -1) + qn
        #             bn += b
        #     dot = torch.mul(a, bn)
        #     dot_sum = torch.sum(dot, dim=1)
        #     # b_norm += dot_sum
        #     nc_loss += dot_sum.mean()
        # return para_cos * nc_loss

    # if diversity_loss_type in ['norm_cos', 'norm_cos1', 'norm_cos2', 'norm_cos3', 'norm_cos4', 'norm_cos5',
    #                            'norm_cos6', 'norm_cos7', 'norm_cos8', 'norm_cos9', 'norm_cos10', 'norm_cos11']: # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos1':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos2':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos3':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos4':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos5':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos6':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos7':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos8':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos9':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos10':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss
    #
    # elif diversity_loss_type == 'norm_cos11':  # 梯度范数加夹角
    #     norm = 0
    #     g = 0
    #     gn = 0
    #     for i in range(args.model_num):
    #         g += x_grads[i] / args.model_num
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         gn += torch.norm(xi, dim=1) / args.model_num
    #
    #     for i in range(args.model_num):
    #         xi = x_grads[i].view(x_grads[i].size(0), -1)
    #         a = torch.norm(xi, dim=1) - gn
    #         # a = torch.mul(tmp, tmp)
    #         b = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 xj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.norm(xj, dim=1) - gn
    #                 b += tmp / args.model_num
    #         proj = torch.mul(a, b)
    #         norm += torch.exp(proj)
    #     norm_loss = torch.log(norm).mean()
    #
    #     cos = 0
    #     qn = g.view(g.size(0), -1)
    #     for i in range(args.model_num):
    #         fi = x_grads[i].view(x_grads[i].size(0), -1)
    #         cos1 = torch.cosine_similarity(fi, qn, dim=1)
    #         cos1 = torch.exp(cos1)
    #         cos2 = 0
    #         for j in range(args.model_num):
    #             if j != i:
    #                 fj = x_grads[j].view(x_grads[j].size(0), -1)
    #                 tmp = torch.cosine_similarity(fj, qn, dim=1)
    #                 cos2 += torch.exp(tmp)
    #         cos += torch.mul(cos1, cos2)
    #     cos_loss = torch.log(cos).mean()
    #     # print('norm_loss:' + str(norm_loss) + 'cos_loss:' + str(cos_loss))
    #
    #     return para_norm * norm_loss + para_cos * cos_loss


def train_loss_ensemble(state, para_a):
    model_list = state['model_list']
    model_name = state['model_name']
    optimzer_list = state['optimzer_list']
    scheduler_list = state['scheduler_list']
    cost = state['cost']
    use_gpu = state['use_gpu']
    global args
    args = state['args']
    data_loader_train = state['data_loader_train']
    data_loader_test = state['data_loader_test']
    model_num = args.model_num
    logging.info('dynamic_type:' + args.dynamic_type)
    best_acc = 0

    # 超参数自适应变化
    if args.loss_type in ['project_lossnew']:
        ce_list = []
        proj_list = []
        cs_list = []
        for epoch in range(args.begin_epoch, args.n_epochs):

            logging.info("Epoch{}/{}".format(epoch, args.n_epochs))
            data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

            for model in model_list:
                model.train()  # 作用是启用batch _normalization 和drop out

            count = 0
            save_count = 0
            for data in data_loader_train:
                if count <= 9:
                    para_a = np.log(1.01 + 0.01 * count)
                    count += 1
                else:
                    count = 0
                    para_a = np.log(1.01)
                # logging.info("train ing")
                x_train, y_train = data
                if use_gpu:
                    x_train, y_train = x_train.cuda(), y_train.cuda()

                x_train_list, y_train_list = [], []
                for i in range(model_num):
                    x_train_list.append(x_train.clone().detach())
                    y_train_list.append(y_train.clone().detach())

                ce_loss = 0
                for i, model in enumerate(model_list):
                    x_train_list[i].requires_grad = True
                    output = model(x_train_list[i])
                    ce_loss += cost(output, y_train_list[i]) / model_num
                    optimzer_list[i].zero_grad()
                # ce_loss.backward(retain_graph=True)
                model_grads = torch.autograd.grad(ce_loss, x_train_list, create_graph=True)

                diversity_loss = 0
                diversity_loss = get_project_loss(list(model_grads), model_list, args.loss_type, para_a)

                x_grads = list(model_grads)
                a = x_grads[0].view(x_grads[0].size(0), -1)
                b = x_grads[1].view(x_grads[1].size(0), -1)
                CS = torch.cosine_similarity(a, b, dim=1)
                cs = CS[0]
                # print('ce_loss' + str(ce_loss) + 'project_loss：' + str(diversity_loss / para_a) + 'CS:' + str(cs) + 'para_a:' + str(para_a))
                (ce_loss + diversity_loss).backward()
                for i, model in enumerate(model_list):
                    optimzer_list[i].step()  # 更新模型

                save_count += 1
                if save_count < 100:
                    ce_list.append(ce_loss.item())
                    proj_list.append(diversity_loss.item() / para_a)
                    cs_list.append(cs.item())
                if (save_count % 100) == 0:
                    ce_list.append(ce_loss.item())
                    proj_list.append(diversity_loss.item() / para_a)
                    cs_list.append(cs.item())

            for i in range(model_num):
                scheduler_list[i].step(epoch)  # 更新学习率

            val_acc = test_loss_ensemble(model_list, optimzer_list, model_name, cost, data_loader_test)

            is_best = val_acc > best_acc
            print('\n' + 'val_acc:', str(val_acc))
            if is_best:
                best_acc = val_acc
            for i in range(model_num):
                if args.para_flag == True:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=str(args.para_config) + '_' + model_name[
                                        i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')
                    pass
                else:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=model_name[
                                               i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')

        text_save('ce_loss', epoch, args.loss_type, ce_list)
        text_save('proj_loss', epoch, args.loss_type, proj_list)
        text_save('cs', epoch, args.loss_type, cs_list)

    elif args.loss_type in ['project_loss',
                            'project_loss_cos',
                            'project_loss_cos1',
                            'project_loss_cosnew',
                            'gal_dis_regular_loss',
                            'project_loss_combine',
                            'project_norm_sum',
                            'project_loss_sort',
                            'GPMR',
                            'VIP'
                            ]:

        for epoch in range(args.begin_epoch, args.n_epochs):

            logging.info("Epoch{}/{}".format(epoch, args.n_epochs))
            data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

            for model in model_list:
                model.train()

            save_count = 0
            for data in data_loader_train:
                x_train, y_train = data
                if use_gpu:
                    x_train, y_train = x_train.cuda(), y_train.cuda()

                x_train_list, y_train_list = [], []
                for i in range(model_num):
                    x_train_list.append(x_train.clone().detach())
                    y_train_list.append(y_train.clone().detach())

                ce_loss = 0
                for i, model in enumerate(model_list):
                    x_train_list[i].requires_grad = True
                    output = model(x_train_list[i])
                    ce_loss += cost(output, y_train_list[i]) / model_num
                    optimzer_list[i].zero_grad()
                # ce_loss.backward(retain_graph=True)
                x_grads = torch.autograd.grad(ce_loss, x_train_list, create_graph=True)

                diversity_loss = get_project_loss_new(list(x_grads), args.loss_type, para_a)

                x_grads = list(x_grads)
                a = x_grads[0].view(x_grads[0].size(0), -1)
                b = x_grads[1].view(x_grads[1].size(0), -1)
                CS = torch.cosine_similarity(a, b, dim=1)
                cs = CS[0]
                # print('ce_loss' + str(ce_loss) + 'project_loss：' + str(diversity_loss / para_a))

                (ce_loss + diversity_loss).backward()

                for i in range(model_num):
                    optimzer_list[i].step()

                # save_count += 1
                # if save_count < 100:
                #     ce_list.append(ce_loss.item())
                #     proj_list.append(diversity_loss.item() / para_a)
                #     cs_list.append(cs.item())
                # if (save_count % 100) == 0:
                #     ce_list.append(ce_loss.item())
                #     proj_list.append(diversity_loss.item() / para_a)
                #     cs_list.append(cs.item())

            for i in range(model_num):
                scheduler_list[i].step(epoch)

            val_acc = test_loss_ensemble(model_list, optimzer_list, model_name, cost, data_loader_test)
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            for i in range(model_num):
                if args.para_flag == True:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=str(args.para_config) + '_' + model_name[
                                        i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')
                    pass
                else:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=model_name[
                                               i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')


    elif args.loss_type in ['norm_cos',
                            'norm_cos1',
                            'norm_cos2',
                            'norm_cos3',
                            'norm_cos4',
                            'norm_cos5',
                            'norm_cos6',
                            'norm_cos7',
                            'norm_cos8',
                            'norm_cos9',
                            'norm_cos10',
                            'norm_cos11'
                            ]:
        for epoch in range(args.begin_epoch, args.n_epochs):
            logging.info("Epoch{}/{}".format(epoch, args.n_epochs))
            data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

            for model in model_list:
                model.train()

            for data in data_loader_train:
                x_train, y_train = data
                if use_gpu:
                    x_train, y_train = x_train.cuda(), y_train.cuda()

                x_train_list, y_train_list = [], []
                for i in range(model_num):
                    x_train_list.append(x_train.clone().detach())
                    y_train_list.append(y_train.clone().detach())

                ce_loss = 0
                ce_list = []
                for i, model in enumerate(model_list):

                    x_train_list[i].requires_grad = True
                    output = model(x_train_list[i])
                    ce_list.append(cost(output, y_train_list[i]))
                    ce_loss += cost(output, y_train_list[i]) / model_num
                    optimzer_list[i].zero_grad()
                # ce_loss.backward(retain_graph=True)

                # x_grads = []
                # for i in args.model_num:
                #     x_grads.append(torch.autograd.grad(ce_list[i], x_train_list[i], create_graph=True))
                x_grads = torch.autograd.grad(ce_loss, x_train_list, create_graph=True)  # x_grade为元组

                diversity_loss = get_project_loss_NC(list(x_grads), args.loss_type, args.para_norm, args.para_cos)
                # print('para_norm:' + str(args.para_norm) + 'para_cos:' + str(args.para_cos))
                # print(ce_loss)
                # print(diversity_loss)

                (ce_loss + diversity_loss).backward()

                for i in range(model_num):
                    optimzer_list[i].step()

            for i in range(model_num):
                scheduler_list[i].step(epoch)

            val_acc = test_loss_ensemble(model_list, optimzer_list, model_name, cost, data_loader_test)
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            for i in range(model_num):
                if args.para_flag == True:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=str(args.para_config) + '_' + model_name[
                                        i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')
                    pass
                else:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=model_name[
                                               i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')

    elif args.loss_type == 'ce':
        for epoch in range(args.begin_epoch, args.n_epochs):
            logging.info("Epoch{}/{}".format(epoch, args.n_epochs))
            data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

            for model in model_list:
                model.train()  # 作用是启用batch _normalization 和drop out
            for data in data_loader_train:  # 每个数据用所有的模型训练一遍
                x_train, y_train = data
                if use_gpu:
                    x_train, y_train = x_train.cuda(), y_train.cuda()

                for i, model in enumerate(model_list):
                    output = model(x_train)
                    _, pred = torch.max(output, 1)
                    loss = cost(output, y_train) / model_num
                    optimzer_list[i].zero_grad()
                    loss.backward()
                    optimzer_list[i].step()

            for i in range(model_num):
                scheduler_list[i].step(epoch)  # 更新学习率
            val_acc = test_loss_ensemble(model_list, optimzer_list, model_name, cost, data_loader_test)
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            for i in range(model_num):
                if args.para_flag == True:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=str(args.para_config) + '_' + model_name[
                                        i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')
                else:
                    save_checkpoint(model_list[i], is_best, best_acc,
                                    save_model_path=args.save_model_path,
                                    suffix=model_name[
                                               i] + '_' + args.dynamic_type + '_' + args.loss_type + '_loss_ensemble_',
                                    filename='checkpoint.pth.tar')


def train_single(state):
    model_list = state['model_list']
    model_name = state['model_name']
    optimzer_list = state['optimzer_list']
    scheduler_list = state['scheduler_list']
    cost = state['cost']
    use_gpu = state['use_gpu']
    args = state['args']
    data_loader_train = state['data_loader_train']
    data_loader_test = state['data_loader_test']
    model_num = args.model_num

    if args.loss_type == 'ce':
        for i, model in enumerate(model_list):
            best_acc = 0
            logging.info("\n model:{}".format(model_name[i]))
            for epoch in range(args.begin_epoch, args.n_epochs):
                model.train()
                running_loss = 0.0
                running_correct = 0
                logging.info("Epoch{}/{}".format(epoch, args.n_epochs))

                data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

                for data in data_loader_train:
                    # logging.info("train ing")
                    x_train, y_train = data
                    if use_gpu:
                        x_train, y_train = x_train.cuda(), y_train.cuda()

                    output = model(x_train)
                    _, pred = torch.max(output, 1)

                    loss = cost(output, y_train)
                    optimzer_list[i].zero_grad()
                    loss.backward()
                    optimzer_list[i].step()

                    running_loss += loss.item()
                    running_correct += torch.sum(pred == y_train)

                scheduler_list[i].step(epoch)
                val_acc = test_single(model, cost, data_loader_test)
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc
                save_checkpoint(model_list[i], is_best, best_acc,
                                save_model_path=args.save_model_path,
                                suffix=model_name[i] + '_single_',
                                filename='checkpoint.pth.tar')

    elif args.loss_type == 'regular_loss':
        for i, model in enumerate(model_list):
            best_acc = 0
            logging.info("\n model:{}".format(model_name[i]))
            for epoch in range(args.begin_epoch, args.n_epochs):
                model.train()
                running_loss = 0.0
                running_correct = 0
                logging.info("Epoch{}/{}".format(epoch, args.n_epochs))

                data_loader_train = tqdm(data_loader_train, desc='Training', ncols=60)

                for data in data_loader_train:
                    # logging.info("train ing")
                    x_train, y_train = data
                    if use_gpu:
                        x_train, y_train = x_train.cuda(), y_train.cuda()

                    x_train.requires_grad = True
                    output = model(x_train)
                    _, pred = torch.max(output, 1)

                    loss = cost(output, y_train)
                    optimzer_list[i].zero_grad()

                    loss.backward(retain_graph=True)
                    x_grads = torch.autograd.grad(loss, x_train, create_graph=True)[0]
                    x_grads = x_grads.view(x_grads.size(0), -1)
                    regular_loss = torch.norm(x_grads, dim=1)
                    regular_loss = regular_loss.mean()
                    regular_loss = regular_loss * 0.5
                    regular_loss.backward()

                    optimzer_list[i].step()

                    running_loss += loss.item()
                    running_correct += torch.sum(pred == y_train)

                scheduler_list[i].step(epoch)
                val_acc = test_single(model, cost, data_loader_test)
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc
                save_checkpoint(model_list[i], is_best, best_acc,
                                save_model_path=args.save_model_path,
                                suffix=model_name[i] + '_' + args.loss_type + '_single_',
                                filename='checkpoint.pth.tar')
