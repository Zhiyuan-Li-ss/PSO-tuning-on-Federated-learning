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

from utils.ga import forest_score, adaption, variation, choose_trees, cross


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
            data_loaded = np.load('forest_ga.npz',allow_pickle=True)
            mean_score_arr = data_loaded['mean_score_arr']
            f_score = data_loaded['f_score']
            forest = data_loaded['forest']     
            best_record = data_loaded['best_record']    
            mean_score_arr = mean_score_arr.tolist()
            f_score = f_score.tolist()
            forest = forest.tolist()
            best_record = best_record.tolist()
            forest_num = len(forest)
            
            
        else:
            mean_score_arr = []
            forest = init()            
            forest_num = len(forest) 
            best_record = []

        
        f_score = [1] * len(forest)
        
        
        
        for i in range(args.ga_epochs):            
            
            f_score = forest_score(args=args,dataset_train=dataset_train,dataset_test=dataset_test,forest=forest)
            ada = adaption(forest,f_score)
            forest = choose_trees(forest, ada)
            forest = cross(forest)
            forest = variation(forest)
            best_record = max(f_score)
            

            
                    
        
            mean_score_arr.append(sum(f_score)/len(f_score))
            print("mean_score:", mean_score_arr)
            print(i, "/", args.ga_epochs, ":")
            print("test_acc:", f_score)
            print("forest:", forest)
            print("best score:",best_record)
            smean_score_arr = np.array(mean_score_arr)
            sf_score = np.array(f_score)
            sforest = np.array(forest)
            sbest_record = np.array(best_record)

            
            np.savez('forest_ga.npz',mean_score_arr=smean_score_arr, f_score=sf_score, forest=sforest, best_record=sbest_record )
            
        
        plt.plot(np.arange(len(mean_score_arr)), mean_score_arr)
        plt.show()
                
        

        
    else: 
        if(args.continue_path):
            
            print('a')
        else:
            print('d')
        
        
        









