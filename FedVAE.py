# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:45:38 2021

@author: Lomo
"""

import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score
import random
import math
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
from copy import deepcopy
import collections
from collections import Counter
from data.dataset import construct_datasets
from data.datasample import Sampleset, MergedSubDataset, SubDataset
import argparse
from model.model import construct_model, construct_vae
from typing import *
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict, Counter
from FedFramework import Client, Server, Controller
from model.VariationalAE import Loss
import matplotlib.pyplot as plt


def split_dataset(datasets:SubDataset, class_num):
    """
    Divide a data set into subsets of data, each containing only one label
    """
    datasets_list = [[] for _ in range(class_num)]
    splitdataset = [None for _ in range(class_num)]
    originalData = datasets.data # list, each element is (Tensor, target)
    for data in originalData:
        datasets_list[data[1]].append(data)
    
    for i in range(class_num):
        if datasets_list[i]:
            splitdataset[i] = SubDataset(datasets_list[i])
    return splitdataset  # each element is "SubDataset" class 
    
# splitdataset = split_dataset(clients0.dataset, 10)


class FedvaeClient(Client):
    def __init__(self, dataset, batch_size, epoch = 10, device = "cpu", validset = None, class_num = 10):
        super(FedvaeClient, self).__init__(dataset=dataset, 
                                           batch_size=batch_size, 
                                           epoch = epoch, 
                                           device = device, 
                                           validset = validset)
        self.batch_size = batch_size
        self.class_num = class_num
        self.vaes = None   # list
        self.vae_state_dicts = None   # list
        self.vae_optimizers = None
        self.vae_epoch = None
        self.vae_loss = None
        self.vae_dataset = split_dataset(self.dataset, self.class_num)  # e.g. [SubDataset, SubDataset, None, None, SubDataset, None, None]
        self.vae_dataloader = [torch.utils.data.DataLoader(self.vae_dataset[i], batch_size=batch_size, shuffle=True, drop_last = True) if self.vae_dataset[i] else None for i in range(self.class_num)] 
        self.owned_class = set(self.dataset.targets)
        self.augmented_data = None  # updated through renew_dataset method
        self.augmented_dataset = None    # updated through renew_dataset method
        self.augmented_dataloader = None # updated through renew_dataset method

    def update_vaes(self, vaes, optimizer_lr, epoch, loss_fn = Loss()):
        """
        these params are allocated from server, this method is called by "Server.distribute" method.
        vaes is a list which include 10(ten classes) vae models
        """
        self.vaes = vaes
        self.vae_optimizers = [torch.optim.Adam(self.vaes[i].parameters(), lr=optimizer_lr, weight_decay=1e-05) for i in range(len(self.vaes))]
        self.vae_epoch = epoch
        self.vae_loss = loss_fn
        
    def _train_vae_once(self):
        """
        Train once for all the generated vae related to their data
        """
        loss_dict = defaultdict(float)
        for i in self.owned_class:
            self.vaes[i].training = True
            self.vaes[i].to(self.device).train()
            train_loss = 0
            for data, _ in self.vae_dataloader[i]:
                 data = data.to(self.device)
                 self.vae_optimizers[i].zero_grad()
                 recon_batch, mu, logvar = self.vaes[i](data)
                 loss = self.vae_loss(recon_batch, data, mu, logvar)
                 loss.backward()
                 self.vae_optimizers[i].step()
                 train_loss += loss.item()
            train_loss /= len(self.vae_dataloader[i].dataset)
            loss_dict[i] = train_loss
        return loss_dict

    
    def train_vae(self):
        loss = defaultdict(list)
        avg_loss = defaultdict(float)
        for e in range(self.vae_epoch):
            loss_dict = self._train_vae_once()
            for key in loss_dict.keys():
                loss[key].append(loss_dict[key])
        self.vae_state_dicts = [self.vaes[i].state_dict() for i in range(self.class_num)]
        for key in loss.keys():
            avg_loss[key] = np.mean(loss[key])
        return loss, avg_loss
                  
    def evaluate_vae(self, num_gen=5, seed = 0, device = "cpu"):
        """
        Note that the vae evaluated here includes only those vae which be trained by self dataset, 
        for example, vae with 0,1,2,3 does not evaluate vae 4,5,6... Only Server will evaluate all vae generators.
        """
        for c in self.owned_class:
            torch.manual_seed(seed)
            latent = torch.randn(num_gen, self.vaes[c].z)
            self.vaes[c] = self.vaes[c].to(device)
            self.vaes[c].eval()
            with torch.no_grad():
                recimg = self.vaes[c].decode(latent)
            
            for i in range(num_gen):
                ax = plt.subplot(num_gen // 10 + 1, num_gen if num_gen<= 10 else 10, i+1)
                img = recimg[i].unsqueeze(0).to(device)
                plt.imshow(img.cpu().squeeze(0).numpy().transpose(1,2,0)) #  cmap='gist_gray'
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == (num_gen if num_gen<= 10 else 10)//2: 
                    ax.set_title('Random Generated Images')
            plt.show()
            
    def renew_dataset(self, device = "cpu"):
        """
        Re-update the "self.dataset" based on its own data distribution so that all classes are in balance
        Because self.vaes defaults to None, "renew_dataset" is premised that self.vaes is not None
        
        Due to torch.randn, "self.augmented_dataset/self.augmented_dataloader" will be varied each time the "renew_dataset" is run
        """
        self.augmented_data = deepcopy(self.dataset.data)
        num_to_generate = []
        data_distribution = Counter(self.dataset.targets)
        max_num = data_distribution.most_common(1)[0][1]
        for c in range(self.class_num):
            num_to_generate.append(max_num - data_distribution[c])
        for c in range(self.class_num):
            if num_to_generate[c] == 0:
                continue
            latent = torch.randn(num_to_generate[c], self.vaes[c].z)
            self.vaes[c] = self.vaes[c].to(device)
            self.vaes[c].eval()
            with torch.no_grad():
                recimg = self.vaes[c].decode(latent)
            for i in range(num_to_generate[c]):
                self.augmented_data.append((recimg[i], c))
        self.augmented_dataset = SubDataset(self.augmented_data)
        self.augmented_dataloader = torch.utils.data.DataLoader(self.augmented_dataset, 
                                                                batch_size=self.batch_size, 
                                                                shuffle=True, 
                                                                drop_last = True)
        
    def _train_classifier_with_augmented_data_once(self):
        self.model.to(self.device).train()
        train_loss = 0
        for data, label in self.augmented_dataloader:
            data = data.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            y = self.model(data)
            loss = self.loss_fn(y, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.augmented_dataloader.dataset)
        return train_loss
    
    def train_classifier_with_augmented_data(self):
        """
        "local_loss" is the loss list recorded in several epochs
        sum(local_loss)/self.epoch is the avg loss
        """
        local_loss = []
        for e in range(self.epoch):
            loss = self._train_classifier_with_augmented_data_once()
            local_loss.append(loss)
        self.model_state_dict = self.model.state_dict()
        return local_loss, sum(local_loss)/self.epoch    
        
            
        

class FedvaeServer(Server):
    def __init__(self, model, vaes, loss_fn, validset, device = "cpu", class_num = 10, vae_loss = Loss()):
        super(FedvaeServer, self).__init__(model=model, 
                                           loss_fn=loss_fn, 
                                           vaildset = validset, 
                                           device = device)
        self.class_num = class_num
        self.vaes = vaes   # list
        self.vae_state_dicts =[self.vaes[i].state_dict() for i in range(self.class_num)]
        self.vae_weight_keys = list(self.vae_state_dicts[0].keys())
        self.vae_loss = vae_loss
        
    def distribute_vaes(self, *clients, epoch, optimizer_lr):
        for client in clients:
            vaes = deepcopy(self.vaes)
            client.update_vaes(vaes, optimizer_lr, epoch, loss_fn = self.vae_loss)
    
    def aggregation_vaes(self, *clients):
        """
        clients : FedvaeClient
        """
        fed_state_dict = [collections.OrderedDict() for _ in range(self.class_num)]
        total_num_data = []  # total_num_data[c] indicates the total number of class c in all clients
        for c in range(self.class_num): # Count the total number of classes in all clients
            num = 0
            for client in clients:
                if client.vae_dataset[c]:              
                    num += client.vae_dataset[c].length
            total_num_data.append(num)
        
        for c in range(self.class_num):    # the c-th genVae
            for key in self.vae_weight_keys:
                key_sum = 0
                for client in clients:
                    if client.vae_dataset[c]:
                        key_sum += (client.vae_dataset[c].length/total_num_data[c]) * client.vae_state_dicts[c][key]
                fed_state_dict[c][key] = key_sum
            try:
                self.vaes[c].load_state_dict(fed_state_dict[c])
            except AttributeError:
                pass
            
    def evaluate_vae(self, num_gen=5, seed = 0, device = "cpu"):
        for c in range(self.class_num):
            torch.manual_seed(seed)
            latent = torch.randn(num_gen, self.vaes[c].z)
            self.vaes[c] = self.vaes[c].to(device)
            self.vaes[c].eval()
            with torch.no_grad():
                recimg = self.vaes[c].decode(latent)
            
            for i in range(num_gen):
                ax = plt.subplot(num_gen // 10 + 1, num_gen if num_gen<= 10 else 10, i+1)
                img = recimg[i].unsqueeze(0).to(device)
                plt.imshow(img.cpu().squeeze(0).numpy().transpose(1,2,0)) #  cmap='gist_gray'
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == (num_gen if num_gen<= 10 else 10)//2: 
                    ax.set_title('Random Generated Images')
            plt.show()  
            
    def save_vae(self, save_path, vae_name):
        for i,vae in enumerate(self.vaes):
            torch.save(vae.state_dict(), os.path.join(save_path, f"{vae_name}_{i}"))
        print("\n vae models have been saved")
    
    
    def load_vae(self, save_path, vae_name):
        for i,vae in enumerate(self.vaes):
            vae.load_state_dict(torch.load(os.path.join(save_path, f"{vae_name}_{i}")))
        print("\n vae models have been loaded")
            
            

            
class FedvaeController(Controller):
    def __init__(self, server:FedvaeServer, clients:List[FedvaeClient]):
         super(FedvaeController, self).__init__(server = server,
                                                clients = clients)         
    
    def _OneCommRound_vae(self, epoch, lr, self_evaluate = False, num_gen = 5):
        # num_gen: how many image to be generated by each client if "self_evaluate = True"
        clients_avgloss = defaultdict(dict)  # dict of dict; dict[dict[int:int]]
        self.server.distribute_vaes(*self.clients, epoch = epoch, optimizer_lr = lr)
        for i in range(self.num_clients):
            loss, avg_loss = self.clients[i].train_vae()
            clients_avgloss[i] = avg_loss
            print(f"Client {i} Finished training VAE \n")
            if self_evaluate:
                self.clients[i].evaluate_vae(num_gen)
        self.server.aggregation_vaes(*self.clients)
        return clients_avgloss

    def start_vae(self, commRound, epoch, lr, self_evaluate = False, num_gen = 5):
        # self_evaluate does not affect the server evaluation after each comm round
        clients_loss_log = []
        for rd in range(1, 1 + commRound):
            print(f"Communication Round: {rd} begin \n")
            clients_avgloss = self._OneCommRound_vae(epoch, lr, self_evaluate = self_evaluate, num_gen = num_gen)
            clients_loss_log.append(clients_avgloss)
            self.server.evaluate_vae(num_gen)
        return clients_loss_log
    
    def save_vae(self, save_path, vae_name):
        self.server.save_vae(save_path, vae_name)
        
    def load_vae(self, save_path, vae_name):
        self.server.load_vae(save_path, vae_name)
        # After the server loads the genVAE, it distributes them to all clients immediately
        self.server.distribute_vaes(*self.clients, epoch = 0, optimizer_lr = 0)
        print("\n vae models have been distributed to clients")
        #for client in self.clients:
        #    client.renew_dataset()
        #print("\n All clients have updated the augmented dataset based on the vae models loaded ")
        
    def _oneCommRound_with_aug(self, epoch, lr, printClassMatrix = True, self_evaluate = True):
        """
        The arguments are the same with "_oneCommRound" method
        """
        acc_dict = defaultdict(list) 
        # acc_dict is used to record the model performance of each client at the end of this round of training
        # acc_dict[i]: [Accuracy, Balanced Accuracy]
        acc_matrix = defaultdict(str) 
        self.server.distribute(*self.clients, epoch = epoch, optimizer_lr = lr)

        [client.renew_dataset() for client in self.clients]
        for i in range(self.num_clients):
            _, loss = self.clients[i].train_classifier_with_augmented_data()  # The only different with "_oneCommRound"
            if self_evaluate:
                print(f"Client {i} Finished: \n")
                acc, balanced_acc, acc_mat = self.clients[i].evaluate(printClassMatrix)
                acc_dict[i] = [acc, balanced_acc]
                acc_matrix[i] = acc_mat
        self.server.aggregation(*self.clients)
        return acc_dict, acc_matrix
    
    def start_with_aug(self, commRound, epoch, lr, printClassMatrix = False, self_evaluate = True):
        """
        The parameter is same as "start"
        """
        print("Initial model performance")
        acc, balanced_acc, acc_mat = self.server.evaluate(printClassMatrix = True)
        
        # "centralAcclist" is used to record the accuracy of the central model on the valid set after each communication round.
        centralAcclist = list(); centralAccmat = list()
        centralAcclist.append([acc, balanced_acc])
        centralAccmat.append(acc_mat)  # NOTE: acc_mat is type of str
        # "clientAccdict" is used to record the accuracy of the client model on the valid set in each communication round.
        # BUT if "self_evaluate" is set to be False, "clientAccdict" will be empty defaultdict(list)
        clientAccdict = None
        clientAccmat = None
        if self_evaluate:
            clientAccdict = defaultdict(list)
            clientAccmat = defaultdict(list)
        for rd in range(1, 1 + commRound):
            print(f"Communication Round: {rd} begin \n")
            
            # if perferd, here could add "if rd == 100: lr = lr * 0.1"
            acc_dict, acc_matrix = self._oneCommRound_with_aug(epoch = epoch,  # # The only different with "start"
                                                               lr = lr, 
                                                               printClassMatrix = printClassMatrix, 
                                                               self_evaluate = self_evaluate)
            if self_evaluate:
                for i in range(self.num_clients):
                    clientAccdict[i].append(acc_dict[i])
                    clientAccmat[i].append(acc_matrix[i])
            
            acc, balanced_acc, acc_mat = self.server.evaluate(printClassMatrix = True)
            centralAcclist.append([acc, balanced_acc])
            centralAccmat.append(acc_mat)
            print(f"Communication Round: {rd} end \n")
        return centralAcclist, centralAccmat, clientAccdict, clientAccmat 
            
    
        

def generateFedvaeClient():
    pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cifar10", required=False, help='cifar10 | cifar100 | mnist | lfw | pubface')
    parser.add_argument('--dataroot', default="E:\Workspace\GDRA\data", required=False, help='path to dataset')
    parser.add_argument('--main-model', type=str, default="LeNetZhu", help='The central model')
    parser.add_argument('--vaez', type=int, default=512, help='z dim of generative model')
    parser.add_argument('--model-seed', type=int, default=1, help='The seed of central model')
    parser.add_argument('--batch', type=int, default=16, help='The batch size set on each client')
    parser.add_argument('--device', type=str, default="cpu", help='The device where FL to be performed')
    parser.add_argument('--channel', type=int, default=3, help='The input data channel')
    parser.add_argument('--saved_path', type=str, default="E:\Workspace\FedVAE\saved_model", help='path that the generative model to be saved')
    parser.add_argument('--saved_name', type=str, default="genVae", help='name of the generative model')
    
    
    opt = parser.parse_args()
    opt.dataset = "mnist"
    opt.dataroot = "E:\Workspace\FedVAE\data"
    opt.transformation = opt.transformation = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.ToTensor()])
    opt.main_model = "LeNetZhu"
    opt.device = "cuda"
    opt.channel = 1
    trainset, validset = construct_datasets(dataset = opt.dataset, data_path = opt.dataroot, transformation=opt.transformation)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=10, num_channels=opt.channel, modelkey = opt.model_seed)
    loss_fn = nn.CrossEntropyLoss()
    ss = Sampleset(trainset)
    

    opt.vaez = 512    
    #single_vae, vae_loss = construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed)
    VAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]
    
    ss = Sampleset(trainset)
    cfg1 = {0:0.25, 1:0.25, 2:0.25, 3:0.25}; num1 = 2000
    cfg2 = {4:0.25, 5:0.25, 6:0.25, 7:0.25}; num2 = 2000
    cfg3 = {8:0.25, 9:0.25, 3:0.25, 6:0.25}; num3 = 2000
    config = [cfg1, cfg2, cfg3]
    nums = [num1, num2, num3]
    
    # build the sub-dataset according to above configuration
    subdataset = []
    for i in range(len(nums)):
        subdataset.append(ss.sampleAccorRatio(dictionary = config[i], num = nums[i], seed = random_seed))
  

    vaeclient0 = FedvaeClient(dataset = subdataset[0], 
                              batch_size = opt.batch, 
                              epoch = None, 
                              device = opt.device, 
                              validset = validset, 
                              class_num = 10)
    vaeclient1 = FedvaeClient(dataset = subdataset[1], 
                              batch_size = opt.batch, 
                              epoch = None, 
                              device = opt.device, 
                              validset = validset, 
                              class_num = 10)
    vaeclient2 = FedvaeClient(dataset = subdataset[2], 
                              batch_size = opt.batch, 
                              epoch = None, 
                              device = opt.device, 
                              validset = validset, 
                              class_num = 10)
    #vaeclient0.update_vaes(VAEs, optimizer_lr=0.01, epoch=100)
    #vaeclient0._train_vae_once()
    #loss, avg_loss = vaeclient0.train_vae()
    
    vaeserver = FedvaeServer(lenet, VAEs, loss_fn, validset, device = opt.device, class_num = 10)
    vaeclients = (vaeclient0, vaeclient1, vaeclient2)
    
    #vaeserver.distribute_vaes(*vaeclients, epoch = 100, optimizer_lr = 0.01)
    #vaeclient0.train_vae()
    #vaeclient1.train_vae()
    
    #vaeserver.aggregation_vaes(vaeclient0)
    #vaeserver.evaluate_vae(5)

    fc = FedvaeController(vaeserver, vaeclients)   
    
    # Start training genVAE and save the trained VAE model
    clients_loss_log = fc.start_vae(commRound=200, epoch = 1, lr = 0.01, self_evaluate = False, num_gen = 5)
    fc.save_vae(save_path = opt.saved_path, vae_name = opt.saved_name)
    
    # Test save and load function of FedvaeServer
    # vaeserver.save_vae(save_path = opt.saved_path, vae_name = opt.saved_name)
    # vaeserver.load_vae(save_path = opt.saved_path, vae_name = opt.saved_name)

    # load the trained VAE model
    VAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]
    fc = FedvaeController(vaeserver, vaeclients)   
    fc.load_vae(save_path = opt.saved_path, vae_name = opt.saved_name)
    
    
    centralAcclist, centralAccmat, clientAccdict, clientAccmat = fc.start_with_aug(commRound = 100, 
                                                                                   epoch = 1, 
                                                                                   lr = 0.01, 
                                                                                   printClassMatrix = False, 
                                                                                   self_evaluate = True)
    
    






