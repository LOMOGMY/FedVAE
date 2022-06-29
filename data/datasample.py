# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:45:57 2021

@author: Lomo
"""


import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
try:
    from dataset import construct_datasets
except:
    from .dataset import construct_datasets
from functools import reduce
import random
import argparse
import math


class SubDataset(Dataset):
    def __init__(self, dataset):
        """
        "dataset" is form of List(Tuple(torch.tensor, int))
        The simplest way to construst dataset is like follows,
        new_data = []
        for i in range(10):
            new_data.append(trainset[i])   # where trainset is a semi-standard Dataset/Imagefolder class
        sd = SubDataset(new_data)
        """
        self.data = dataset    #  List type  
        self.length = len(dataset)
        self.targets = [self.data[i][1] for i in range(self.length)]   # List type
    
    def __getitem__(self, index):
        x, y =self.data[index]
        return x, y
    
    def __len__(self):
        return self.length
    

class MergedSubDataset(Dataset):
    """
    This class take several "SubDataset" in and merge them into a new dataset,
    All methods and attribute are remained the same with "SubDataset"
    
    Example: 
        sd_list = (sd1, sd2, sd3)  # sd1, sd2, sd3 are "SubDataset" type
        msd = MergedSubDataset(*sd_list)
    """
    def __init__(self, *SubDataset):
        self.data = reduce(lambda x,y: x+y, [sub.data for sub in SubDataset])  # In order
        self.length = sum(len(sub) for sub in SubDataset)
        self.targets = reduce(lambda x,y: x+y, [sub.targets for sub in SubDataset])  # In order
        
    def __getitem__(self, index):
        x, y =self.data[index]
        return x, y
    
    def __len__(self):
        return self.length
        
    

class Sampleset:
    """
    Given a dataset (Can be form as torchvision.datasets.mnist.MNIST, ImageFolder), 
    this class can sample the data to form a new sub-dataset (generate class SubDataset)
    
    """
    def __init__(self, dataset, valset = None):
        """
        dataset class is from "construct_dataloaders"/[RENAME TO "construct_datasets"]; 
        See data.dataset.construct_dataloaders/[data.dataset.construct_datasets]
            For example:
            trainset, valset = construct_dataloaders(dataset = ..., data_path = ..., transformation = ...)
            "trainset" here could be dataset and "valset" could be valset
        """
        self.dataset = dataset
        self.valset = valset
        self.class_idx =  None  # Type:defaultdict;  key: class num;  value: List[int]. belong to self.dataset
        self.class_idx_val = None    # belong to self.valset
        self.build_class_idx(train = True, val = False)
        
    def sampleAccorNum(self, dictionary, from_where = "dataset", seed = 0):
        """
        "dictionary" is form of dict and "key" is class num (int type) and "value" is the number of sample to be chosen
        "from_where" tell where to sample, could be either "dataset" or "valset"
        For example:
            dictionary = {0:161, 1:322, 2:211, 3:107}
        
        NOTE: To avoid double sampling, the sampled index will removed from the self.class_idx after each sampling
        """
        random.seed(seed)
        sample_idx = []
        if from_where == "dataset":
            for key, val in dictionary.items():
                samp = random.sample(self.class_idx[key], val)
                [self.class_idx[key].remove(i) for i in samp]
                sample_idx.extend(samp)
            
            random.shuffle(sample_idx)
            
            dataset_list = []  # will be passed to "SubDataset" class
            for i in sample_idx:
                dataset_list.append(self.dataset[i])
       
            return SubDataset(dataset_list)
   
        if from_where == "valset":
            pass

        return 
    
    def sampleAccorRatio(self, dictionary, num, from_where = "dataset", seed = 0):
        """
        "dictionary" is form of dict and "key" is class num (int type) and "value" is the ratio of sample to be chosen.
        The sum of ratio demand to be less or equals to one. If less than one, the sampled data will be less than "num".
        "num" denote the total number of sample to be chosen
        For example:
            dictionary = {0:1/5, 1:1/10, 5:1/4}
            num = 100
            
        NOTE: To avoid double sampling, the sampled index will removed from the self.class_idx after each sampling
        """
        assert num <= len(self.dataset) if from_where == "dataset" else num <= len(self.valset)
        dic = defaultdict(int)
        for key, val in dictionary.items():
            dic[key] = round(val * num)
        
        return self.sampleAccorNum(dic, from_where = from_where, seed = seed)
            
        
    @staticmethod
    def cal_distribution(dataset):
        """
        Parameters
        ----------
        dataset : TYPE: SubDataset / torchvision.datasets.mnist.MNIST / ImageFolder / MergedSubDataset
        
        Returns
        -------
        c : TYPE: collections.Counter
            key: class; value: num of that class
        ratio_dict : dict
            key: class; value: ratio of that class
        """
        n = len(dataset); ratio_dict = defaultdict(float)
        try:
            c = Counter(dataset.targets.tolist())
        except:
            c = Counter(dataset.targets)
            
        for key in c.keys():
            ratio_dict[key] = c[key]/n
        return c, ratio_dict
    
    def build_class_idx(self, train = True, val = False):
        """
        Build a dict which key denotes the class and value denotes the index list corspd to that class.
        If the dict has been clipped by sampleAccorNum/sampleAccorRatio method, you can run the method again to REBUILD 
        the dictionary self.class_idx/ self.class_idx_val
        Returns
        -------
        defaultdict{int: List(int)}.  
        """
        dic_train = defaultdict(list)
        dic_val = defaultdict(list)
        if train:
            idx = self.dataset.targets
            if not isinstance(idx, torch.Tensor):
                idx = torch.Tensor(self.dataset.targets)
            for i in self.dataset.class_to_idx.values():
                dic_train[i] = torch.where(idx == i)[0].tolist()
            self.class_idx = dic_train
        if val:
            idx = self.valset.targets
            if not isinstance(idx, torch.Tensor):
                idx = torch.Tensor(self.dataset.targets)
            for i in self.valset.class_to_idx.values():
                dic_val[i] = torch.where(idx == i)[0].tolist()   

            self.class_idx_val = dic_val
        
# Multiclass Imbalance Degree
def cal_MID(counter, ratio_dict, class_num):
    """
    Parameters
    ----------
    counter : TYPE: collections.Counter
        the first output of Sampleset.cal_distribution; e.g. Counter({5: 28, 1: 8, 0: 4})
    ratio_dict : dict
        the first output of Sampleset.cal_distribution; e.g. defaultdict(float, {5: 0.7, 1: 0.2, 0: 0.1})
    class_num : TYPE
        The total number of class, e.g. 10 for MNIST

    Returns: TYPT: int
    -------
    Imbalance Degree (0 for strictly balanced dataset)

    """
    MID = 0
    N = sum(counter.values())
    for i in range(class_num):
        if counter[i] == 0:
            continue
        MID += counter[i] * math.log10(N/(class_num * counter[i]))
    MID = (math.log10(class_num)/N) * MID
    return MID
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="mnist", required=False, help='cifar10 | cifar100 | mnist | lfw | pubface')
    parser.add_argument('--dataroot', default="E:\Workspace\FedVAE\data", required=False, help='path to dataset')
    parser.add_argument('--seed', type=int, default=1, help='The seed of sampling data')
    parser.add_argument('--config', type=list, default=None, help='configuration of different subdataset, TYPE: LIST[DICT]')
    parser.add_argument('--nums', type=list, default=None, help='sample num of different subdataset, TYPE: LIST[int]')
    parser.add_argument('--transformation', default=None, help='transformation of the original dataset')
    opt = parser.parse_args()
    
    opt.dataset = "mnist"
    opt.dataroot = "E:\Workspace\FedVAE\data"
    opt.transformation = transforms.Compose([transforms.Resize(32),
                                         transforms.ToTensor()])
    
    trainset, validset = construct_datasets(dataset = opt.dataset, 
                                            data_path = opt.dataroot, 
                                            transformation=opt.transformation)
    
    cfg1 = {0:0.1, 1:0.2, 5:0.7}; num1 = 40
    cfg2 = {3:0.4, 6:0.2, 5:0.4}; num2 = 30
    cfg3 = {1:0.5, 9:0.3, 8:0.2}; num3 = 60
    
    opt.config = [cfg1, cfg1, cfg3]
    opt.nums = [num1, num2, num3]
    
    ss = Sampleset(trainset)
    subdataset = []
    for i in range(len(opt.nums)):
        subdataset.append(ss.sampleAccorRatio(dictionary = opt.config[i], num = opt.nums[i], seed = 1))
        
    msd = MergedSubDataset(*subdataset)


    dl = torch.utils.data.DataLoader(msd, batch_size = 4, shuffle = True)

    # Before proceeding to the next experiment, resampling, the following method needs to be implemented
    ss.build_class_idx()
    
    
    # test MID
    class_num = len(ss.class_idx.keys())
    cfg1 = {0:0.25, 1:0.25, 2:0.25, 3:0.25}; num1 = 40
    data1 = ss.sampleAccorRatio(dictionary = cfg1, num = num1, seed = 1)
    counter, ratio_dict = ss.cal_distribution(data1)
    cal_MID(counter, ratio_dict, class_num)  # -0.3979400086720376
    
    
    
    
    
    
    