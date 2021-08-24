import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

from utils.ga import forest_score, update_f


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)


    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

                                        
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    

    if args.ga:

        def init():
            if args.model == 'cnn' and args.dataset == 'cifar':
                forest = []
                hidden_1 = [10,100,3]
                hidden_2 = [20,140,3]
                lr = [1,21,5]
                momentum = [1,9,4]
    
                local_epochs = [3,15,3]
                fraction = [1,3,2]
                forest_num = fraction[2]*hidden_2[2]*lr[2]*momentum[2]*local_epochs[2]*hidden_1[2]
                print('the number of the forests:', forest_num)
                for i in range(forest_num):

                    para_1 = int(random.random() * (hidden_1[1] - hidden_1[0]) + hidden_1[0])
                    para_2 = int(random.random() * (hidden_2[1] - hidden_2[0]) + hidden_2[0])
                    para_3 = random.random() * (lr[1] - lr[0]) + lr[0]
                    para_4 = random.random() * (momentum[1] - momentum[0]) + momentum[0]
                    para_5 = int(random.random() * (local_epochs[1] - local_epochs[0]) + local_epochs[0])
                    para_6 = random.random() * (fraction[1] - fraction[0]) + fraction[0]
                    
                    forest.append([para_1,para_2,para_3/1000,para_4/10,para_5,para_6/10])
        
                return forest
                
                            
                
            
            else:
                forest = []
                hidden_1 = [10,100,2]
                lr = [1,21,2]
                momentum = [1,9,2]    
                local_epochs = [3,7,2]
                fraction = [1,3,2]
                forest_num = fraction[2]*lr[2]*momentum[2]*local_epochs[2]*hidden_1[2]
                print('the number of the forests:', forest_num)
                for j in range(forest_num):
                    para_1 = int(random.random() * (hidden_1[1] - hidden_1[0]) + hidden_1[0])
                    para_3 = random.random() * (lr[1] - lr[0]) + lr[0]
                    para_4 = random.random() * (momentum[1] - momentum[0]) + momentum[0]
                    para_5 = int(random.random() * (local_epochs[1] - local_epochs[0]) + local_epochs[0])
                    para_6 = random.random() * (fraction[1] - fraction[0]) + fraction[0]
                    
                    forest.append([para_1,para_3/1000,para_4/10,para_5,para_6/10])
                return forest
        
        if args.continue_path:
            data_loaded = np.load('forest.npz',allow_pickle=True)
            mean_score_arr = data_loaded['mean_score_arr']
            f_score = data_loaded['f_score']
            forest = data_loaded['forest']
            pbest_score = data_loaded['pbest_score']
            pbest_para = data_loaded['pbest_para']
            gbest_score = data_loaded['gbest_score']
            gbest_para = data_loaded['gbest_para']
            forest_dir = data_loaded['forest_dir']
            gbest_record = data_loaded['gbest_record']
            gbest_record = gbest_record.tolist()
            mean_score_arr = mean_score_arr.tolist()
            f_score = f_score.tolist()
            forest = forest.tolist()
            pbest_score = pbest_score.tolist()
            pbest_para = pbest_para.tolist()
            gbest_score = gbest_score.tolist()
            gbest_para = gbest_para.tolist()
            forest_dir = forest_dir.tolist()
            forest_num = len(forest)
            
            
        else:
            mean_score_arr = []
            forest = init()
            forest_dir = []
            forest_num = len(forest)
            for i in range(forest_num):
                forest_dir_temp = []
                for j in range(len(forest[0])):
                    forest_dir_temp.append(random.random())
                    
                forest_dir.append(forest_dir_temp)
            
            gbest_para = []
            gbest_score = 0
            pbest_para = [0]*forest_num
            pbest_score = [0]*forest_num
            gbest_record = []
        c1 = 2
        c2 = 2
        
       
        
        
        
        for i in range(args.ga_epochs):
            
            f_score = forest_score(args=args,dataset_train=dataset_train,dataset_test=dataset_test,forest=forest)
            for j in range(forest_num):
                if pbest_score[j]<f_score[j]:
                    pbest_score[j] = f_score[j]
                    pbest_para[j] = forest[j]
            if max(f_score)>gbest_score:
                gbest_score = max(f_score)
                gbest_para = forest[f_score.index(max(f_score))]
            gbest_record.append(gbest_score)
            forest, forest_dir = update_f(args, forest, gbest_para, pbest_para, forest_dir, c1, c2, 50, i)

            
                    
        
            mean_score_arr.append(sum(f_score)/len(f_score))
            print("mean_score:", mean_score_arr)
            print(i, "/", args.ga_epochs, ":")
            print("test_acc:", f_score)
            print("forest:", forest)
            print("p_best_score:",pbest_score)
            print("p_best_para:",pbest_para)
            print("gbest_score:",gbest_score)
            print("gbest_para:",gbest_para)
            print("forest_dir:",forest_dir)
            print("gbest_record:", gbest_record)
            smean_score_arr = np.array(mean_score_arr)
            sf_score = np.array(f_score)
            sforest = np.array(forest)
            spbest_score = np.array(pbest_score)
            spbest_para = np.array(pbest_para)
            sgbest_score = np.array(gbest_score)
            sgbest_para = np.array(gbest_para)
            sforest_dir = np.array(forest_dir)
            sgbest_record = np.array(gbest_record)
            
            np.savez('forest.npz',mean_score_arr=smean_score_arr, f_score=sf_score, forest=sforest, pbest_score=spbest_score,pbest_para=spbest_para, gbest_score=sgbest_score, gbest_para=sgbest_para, forest_dir=sforest_dir, gbest_record=sgbest_record )
            
        
        plt.plot(np.arange(len(mean_score_arr)), mean_score_arr)
        plt.show()
                
        

        
    else: 
        if(args.continue_path):
            
            print('a')
        else:
            print('d')
        