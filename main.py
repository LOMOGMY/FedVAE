# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:48:42 2021

@author: Lomo
"""

import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pickle
import numpy as np
from data.dataset import construct_dataloaders
from model.model import construct_model
import argparse
from FedFramework import Client, Server, Controller
from data.datasample import Sampleset, MergedSubDataset, SubDataset, cal_MID

def generate_clients(ss, opt, validset, cfg, nums, seed=1):
    # ss: Sampleset
    subdataset = []
    clients = []
    for i in range(len(nums)):
        subdataset.append(ss.sampleAccorRatio(dictionary = cfg[i], num = nums[i], seed = seed)) 
        clients.append(Client(dataset = subdataset[i], batch_size = opt.batch, epoch = None, device = opt.device, validset = validset))
    
    return subdataset, clients

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="cifar10", required=False, help='cifar10 | cifar100 | mnist | lfw | pubface')
    parser.add_argument('--dataroot', default="E:\Workspace\GDRA\data", required=False, help='path to dataset')
    parser.add_argument('--main-model', type=str, default="LeNetZhu", help='The central model')
    parser.add_argument('--model-seed', type=int, default=1, help='The seed of central model')
    parser.add_argument('--device', type=str, default="cpu", help='The device where FL to be performed')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--channel', type=int, default=3)
    
    opt = parser.parse_args()
    opt.dataset = "mnist"
    opt.dataroot = "E:\Workspace\FedVAE\data"
    opt.transformation = opt.transformation = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.ToTensor()])
    opt.device = "cuda"
    opt.channel = 1
    opt.main_model = "LeNetZhu"
    
    trainset, validset = construct_dataloaders(dataset = opt.dataset, data_path = opt.dataroot, transformation=opt.transformation)
    lenet, random_seed = construct_model(model="LeNetZhu", num_classes=10, num_channels=1, modelkey = opt.model_seed)
    loss_fn = nn.CrossEntropyLoss()
    ss = Sampleset(trainset)
    
    
    # First configuration
    cfg1 = {0:0.1, 1:0.2, 2:0.7}; num1 = 400
    cfg2 = {0:0.2, 1:0.7, 2:0.1}; num2 = 400
    cfg3 = {0:0.7, 1:0.1, 2:0.2}; num3 = 400
    config = [cfg1, cfg1, cfg3]
    nums = [num1, num2, num3]
    
    server = Server(lenet, loss_fn, validset, device = opt.device)
    subdataset, clients = generate_clients(ss, opt, validset, config, nums)
    msd = MergedSubDataset(*subdataset)
    central_client = Client(dataset = msd, batch_size = opt.batch, epoch = None, device = opt.device, validset = validset)
    server.distribute(central_client, epoch = 100, optimizer_lr = 0.01)
    
    
    counter, ratio_dict = ss.cal_distribution(subdataset[0])
    cal_MID(counter, ratio_dict, len(ss.class_idx.keys()))   # -0.651774627142776
    # Central model
    _, loss = central_client.train_classifier()
    central_acc, central_balanced_acc, acc_mat = central_client.evaluate(printClassMatrix = True) # (0.3119, 0.29728451487141316)
    
    # Fed model
    controller = Controller(server, clients)
    # res_dict = controller._oneCommRound(epoch = 10, lr = 0.01, printClassMatrix = False, self_evaluate = True)
    centralAcclist, centralAccmat, clientAccdict, clientAccmat = controller.start(commRound = 20, 
                                                                                  epoch = 50, 
                                                                                  lr = 0.01, 
                                                                                  printClassMatrix = False, 
                                                                                  self_evaluate = True)
    # (0.3117, 0.29712007628581605)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    















