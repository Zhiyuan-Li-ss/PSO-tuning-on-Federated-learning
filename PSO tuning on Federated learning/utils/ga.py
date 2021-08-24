# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:55:23 2021

@author: 99554
"""
from models.Update import LocalUpdate
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser

from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


VAR_P = 0.4

def forest_score(args,dataset_train,dataset_test,forest):
    img_size = dataset_train[0][0].shape
    score = []
    for tree_idx in range(len(forest)):
        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args,n_hidden_1= int(forest[tree_idx][0]), n_hidden_2= int(forest[tree_idx][1])).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args,n_hidden= int(forest[tree_idx][0])).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden= int(forest[tree_idx][0]), dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train()
        
        if args.dataset == 'mnist':
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            
        loss_train = []    
        net_int = copy.deepcopy(net_glob)
        w_gint = net_int.state_dict()
        print(forest[tree_idx])
        if args.all_clients: 
            print("Aggregation over all clients")
            w_locals = [w_gint for i in range(args.num_users)]
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(forest[tree_idx][-1] * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],local_ep= int(forest[tree_idx][-2]), lr = forest[tree_idx][-4], momentum = forest[tree_idx][-3])
                w, loss = local.train(net=copy.deepcopy(net_int).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_gint = FedAvg(w_locals)
            # copy weight to net_glob
            net_int.load_state_dict(w_gint)
    
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Tree {:3d} Round {:3d}, Average loss {:.3f}'.format(tree_idx, iter, loss_avg))
            loss_train.append(loss_avg)
 
           
        net_int.eval()           
        acc_train, loss_train = test_img(net_int, dataset_train, args)
        acc_test, loss_test = test_img(net_int, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        score_test = acc_test.item()
        score.append(score_test)


    
    print(score)       
    return score


def update_f(args, forest, gbest_para, pbest_para, forest_dir, c1, c2, max_iter, current_iter):
    if args.model == 'cnn' and args.dataset == 'cifar':
        up = [70,100,0.2,0.9,15,0.6]
        down = [10,20,0,0,1,0]
    else:
        up = [100,0.21,0.9,7,0.3]
        down = [10,0.01,0.1,3,0.1]
        
    for i in range(len(forest)):
        w = (0.9 - 0.4)*(max_iter- current_iter)/ max_iter + 0.4
        a1 = [x * w  for x in forest_dir[i]]
        a2 = [y * c1 * random.random() for y in list(np.array(pbest_para[i]) - np.array(forest[i]))]
        a3 = [z * c2 * random.random() for z in list(np.array(gbest_para) - np.array(forest[i]))]
        forest_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
    #            particle_dir[i] = self.w * particle_dir[i] + self.c1 * random.random() * (pbest_parameters[i] - particle_loc[i]) + self.c2 * random.random() * (gbest_parameter - particle_dir[i])
        forest[i] = list(np.array(forest[i]) + np.array(forest_dir[i]))
    parameter_list = []
    for i in range(len(forest[0])):
        tmp1 = []
        for j in range(len(forest)):
            tmp1.append(forest[j][i])
        parameter_list.append(tmp1)
    ### 2.2 每个参数取值的最大值、最小值、平均值   
    value = []
    for i in range(len(forest[0])):
        tmp2 = []
        tmp2.append(max(parameter_list[i]))
        tmp2.append(min(parameter_list[i]))
        value.append(tmp2)
        
    for i in range(len(forest)):
        for j in range(len(forest[0])):
            forest[i][j] = (forest[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (up[j] - down[j]) + down[j]

        forest[i][0] = int(forest[i][0])
        forest[i][-2] = int(forest[i][-2])
        forest[i][-5] = int(forest[i][-5])
        
        
        
        
        
        
    return forest, forest_dir

    
def adaption(forest,score):
    best_pos = np.argmax(score)
    global BEST_TREE
    BEST_TREE = copy.deepcopy(forest[best_pos])
    sm = np.sum(score)
    ada = score / sm
    for i in range(1, len(ada)):
        
        ada[i] = ada[i] + ada[i - 1]
        
    return ada


def choose_trees(forest, ada):
    sz = len(forest)
    result = []
    for i in range(sz):
        r = random.random()
        for j in range(len(ada)):
            if r <= ada[j]:
                result.append(copy.deepcopy(forest[j]))
                break
    return result
        
    
def _cross_2_tree(t1, t2):
    sz = len(t1)
    pos = random.randint(0, sz - 1)
    p1 = random.random()
    p2 = random.random()
    p3 = random.random()
    p4 = random.random()
    p5 = random.random()
    if pos==0:
        n11 = int(t1[0] + p1 * (t2[0]-t1[0]))
        n12 = int(t2[0] + p1 * (t1[0]-t2[0]))
        t1[pos] = n11
        t2[pos] = n12
    if pos==1:
        n21 = t1[1] + p2 * (t2[1]-t1[1])
        n22 = t2[1] + p2 * (t1[1]-t2[1])
        t1[pos] = n21
        t2[pos] = n22
    if pos==2:
        n31 = t1[2] + p3 * (t2[2]-t1[2])
        n32 = t2[2] + p3 * (t1[2]-t2[2])
        t1[pos] = n31
        t2[pos] = n32
    if pos==3:
        n41 = int(t1[3] + p4 * (t2[3]-t1[3]))
        n42 = int(t2[3] + p4 * (t1[3]-t2[3]))
        t1[pos] = n41
        t2[pos] = n42
    if pos==4:
        n51 = t1[4] + p3 * (t2[4]-t1[4])
        n52 = t2[4] + p3 * (t1[4]-t2[4])
        t1[pos] = n51
        t2[pos] = n52
        

    return [t1, t2]


def cross(forest):
    result = []
    sz = len(forest)
    for i in range(1, sz, 2):
        result.extend(_cross_2_tree(forest[i - 1], forest[i]))
    return result


def variation(forest):
    result = []
    sz = len(forest[0])
    print(sz)
    for i in range(len(forest)):
        r = random.random()
        if r < VAR_P:
            result.append(forest[i])
            continue

        # 变异
        
        pos = random.randint(0, 4)
        up = random.random()

        if up > 0.5:
            forest[i][pos] = forest[i][pos] * up
        else:
            forest[i][pos] = forest[i][pos] * (1+up)

        if pos==0 or pos==3:
            forest[i][pos] = int(forest[i][pos])
        result.append(forest[i])
    return result
    

    