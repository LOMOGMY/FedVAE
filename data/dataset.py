# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:56:03 2021

@author: Lomo
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from os import listdir
from os.path import join
from os.path import basename
from PIL import Image
from torchvision.datasets.folder import ImageFolder

#default_transform = transforms.Compose([transforms.ToTensor(),
#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])

#transform = transforms.Compose([
#           transforms.Resize(size),
#           transforms.CenterCrop(size),
#           transforms.ToTensor(),
#           transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])



default_transform = transforms.Compose([transforms.ToTensor()])

# data_path = "E:\Workspace\GDRApro\data"
def build_MNIST(data_path, transformation = None):
    
    if not transformation:
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=default_transform)
        validset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=default_transform)
        
    else:
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transformation)
        validset = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transformation)
        
    return trainset, validset

# data_path = "E:\Workspace\GDRA\data"
def build_CIFAR10(data_path, transformation = None):
    if not transformation:
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=default_transform)
        validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=default_transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transformation)
        validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transformation)
        
    return trainset, validset
        
# data_path = "E:\Workspace\GDRA\data"
def build_CIFAR100(data_path, transformation = None):
    if not transformation:
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=default_transform)
        validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=default_transform)
    else:
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transformation)
        validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transformation)

    return trainset, validset      

# data_path = "E:\Workspace\GDRA\data\lfw"    
def build_lfw(data_path, transformation = None):
    if not transformation:
        trainset = ImageFolder(data_path+"\\train", transform=default_transform)
        validset = ImageFolder(data_path+"\\test", transform=default_transform)
    else:
        trainset = ImageFolder(data_path+"\\train", transform=transformation)
        validset = ImageFolder(data_path+"\\test", transform=transformation)    
    
    return trainset, validset

def build_original_lfw(data_path, transformation = None):
    if not transformation:
        trainset = ImageFolder(data_path, transform=default_transform)
        #validset = ImageFolder(data_path, transform=default_transform)
    else:
        trainset = ImageFolder(data_path, transform=transformation)
        #validset = ImageFolder(data_path, transform=transformation)    
    
    return trainset, None

# data_path = "E:\Workspace\GDRA\data\pubface"  
def build_pubface(data_path, transformation = None):
    if not transformation:
        trainset = ImageFolder(data_path+"\\train", transform=default_transform)
        validset = ImageFolder(data_path+"\\test", transform=default_transform)
    else:
        trainset = ImageFolder(data_path+"\\train", transform=transformation)
        validset = ImageFolder(data_path+"\\test", transform=transformation)    
    
    return trainset, validset    

def build_original_pubface(data_path, transformation = None):
    if not transformation:
        trainset = ImageFolder(data_path, transform=default_transform)
    else:
        trainset = ImageFolder(data_path, transform=transformation)
    
    return trainset, None  


def construct_dataloaders(dataset: str, data_path: str, transformation):
    if dataset in ["MNIST", "mnist"]:
        trainset, validset = build_MNIST(data_path, transformation = transformation)
        return trainset, validset
    
    if dataset in ["CIFAR10", "cifar10"]:
        trainset, validset = build_CIFAR10(data_path, transformation = transformation)
        return trainset, validset
    
    if dataset in ["CIFAR100", "cifar100"]:
        trainset, validset = build_CIFAR100(data_path, transformation = transformation)
        return trainset, validset  
    
    if dataset in ["lfw", "LFW"]:
        trainset, validset = build_lfw(data_path, transformation = transformation)
        return trainset, validset
    
    if dataset in ["lfw_original"]:
        trainset, validset = build_original_lfw(data_path, transformation = transformation)
        return trainset, validset
    
    if dataset in ["pubface", "PUBFACE"]:
        trainset, validset = build_pubface(data_path, transformation = transformation)
        return trainset, validset    

    if dataset in ["pubface_original"]:
        trainset, validset = build_original_pubface(data_path, transformation = transformation)
        return trainset, validset

def construct_datasets(dataset: str, data_path: str, transformation):
    # Rename of "construct_dataloaders"
    return construct_dataloaders(dataset, data_path, transformation)





        
        
    
