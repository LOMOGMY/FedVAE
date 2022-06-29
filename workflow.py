# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:14:27 2021

@author: Lomo
"""

"""
This is the whole Fedvae experimental flow sample code
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
from FedVAE import FedvaeClient, FedvaeServer, FedvaeController

def generate_clients(ss:Sampleset, opt, validset, cfg, nums, seed=1):
    # ss: Sampleset
    subdataset = []
    clients = []
    for i in range(len(nums)):
        subdataset.append(ss.sampleAccorRatio(dictionary = cfg[i], num = nums[i], seed = seed)) 
        clients.append(FedvaeClient(dataset = subdataset[i], 
                                    batch_size = opt.batch, 
                                    epoch = None, 
                                    device = opt.device, 
                                    validset = validset,
                                    class_num = opt.class_num))
    return clients, subdataset


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
    parser.add_argument('--class_num', type=int, default=10, help='How many classes are on the data set')
    parser.add_argument('--central_epoch', type=int, default=100, help='The epoch num that central training need to obey')
    parser.add_argument('--epoch', type=int, default=1, help='The epoch num that FL clients training need to obey')
    parser.add_argument('--commRound', type=int, default=100, help='The communication round')

    opt = parser.parse_args()
    
    run_on_pc = True
    if run_on_pc:
        opt.dataset = "mnist"
        opt.dataroot = "E:\Workspace\FedVAE\data"
        opt.saved_path = "E:\Workspace\FedVAE\saved_model"
        opt.saved_name = "generativeVAE"
        opt.transformation = opt.transformation = transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.ToTensor()])
        opt.main_model = "LeNetZhu"
        opt.device = "cuda"
        opt.channel = 1
        opt.vaez = 512
    trainset, validset = construct_datasets(dataset = opt.dataset, data_path = opt.dataroot, transformation=opt.transformation)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=10, num_channels=opt.channel, modelkey = opt.model_seed)
    initVAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]
    loss_fn = nn.CrossEntropyLoss()
    ss = Sampleset(trainset)
        
    # Configure client information
    cfg0 = {0:1}; num0 = 1000
    cfg1 = {1:1}; num1 = 1000
    cfg2 = {2:1}; num2 = 1000
    cfg3 = {3:1}; num3 = 1000
    cfg4 = {4:1}; num4 = 1000
    cfg5 = {5:1}; num5 = 1000
    cfg6 = {6:1}; num6 = 1000
    cfg7 = {7:1}; num7 = 1000
    cfg8 = {8:1}; num8 = 1000
    cfg9 = {9:1}; num9 = 1000
    config = [cfg0, cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7, cfg8, cfg9]
    nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]
    
    
    # performance of central training
    ss.build_class_idx()
    clients, subdataset = generate_clients(ss, opt, validset, config, nums, seed=random_seed)
    msd = MergedSubDataset(*subdataset)
    server = FedvaeServer(lenet, initVAEs, loss_fn, validset, device = opt.device, class_num = opt.class_num, vae_loss = Loss())
    central_client = FedvaeClient(dataset = msd, 
                                  batch_size = opt.batch, 
                                  epoch = None, 
                                  device = opt.device, 
                                  validset = validset,
                                  class_num = opt.class_num)
    
    server.distribute(central_client, epoch = opt.central_epoch, optimizer_lr = 0.01)
    _, loss = central_client.train_classifier()
    central_acc, central_balanced_acc, acc_mat = central_client.evaluate(printClassMatrix = True) 
    # 0.9736
    
    
    # performance of Fed learning
    ss.build_class_idx()
    clients, subdataset = generate_clients(ss, opt, validset, config, nums, seed=random_seed)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=opt.class_num, num_channels=opt.channel, modelkey = opt.model_seed)
    initVAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]
    
    server = FedvaeServer(lenet, initVAEs, loss_fn, validset, device = opt.device, class_num = opt.class_num, vae_loss = Loss())

    controller = FedvaeController(server, clients)
    centralAcclist, centralAccmat, clientAccdict, clientAccmat = controller.start(commRound = opt.commRound, 
                                                                                  epoch = opt.epoch, 
                                                                                  lr = 0.01, 
                                                                                  printClassMatrix = False, 
                                                                                  self_evaluate = True)
    # 0.1
    
    # performance of FedVAE
    ss.build_class_idx()
    clients, subdataset = generate_clients(ss, opt, validset, config, nums, seed=random_seed)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=opt.class_num, num_channels=opt.channel, modelkey = opt.model_seed)
    initVAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]     
    server = FedvaeServer(lenet, initVAEs, loss_fn, validset, device = opt.device, class_num = opt.class_num, vae_loss = Loss())
    
    fc = FedvaeController(server, clients)   
    
    # Start training genVAE and save the trained VAE model
    clients_loss_log = fc.start_vae(commRound=200, epoch = opt.epoch, lr = 0.01, self_evaluate = False, num_gen = 5)
    fc.save_vae(save_path = opt.saved_path, vae_name = opt.saved_name)    

    # load the trained VAE model
    ss.build_class_idx()
    clients, subdataset = generate_clients(ss, opt, validset, config, nums, seed=random_seed)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=opt.class_num, num_channels=opt.channel, modelkey = opt.model_seed)
    initVAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]     
    server = FedvaeServer(lenet, initVAEs, loss_fn, validset, device = opt.device, class_num = opt.class_num, vae_loss = Loss())
    
    VAEs = [construct_vae(num_channels=opt.channel, z = opt.vaez, modelkey = opt.model_seed) for _ in range(10)]
    fc = FedvaeController(server, clients)   
    fc.load_vae(save_path = opt.saved_path, vae_name = opt.saved_name)
    
    
    centralAcclist, centralAccmat, clientAccdict, clientAccmat = fc.start_with_aug(commRound = 100, 
                                                                                   epoch = 1, 
                                                                                   lr = 0.01, 
                                                                                   printClassMatrix = False, 
                                                                                   self_evaluate = True)
    
    # mean -> 0.95
   


















