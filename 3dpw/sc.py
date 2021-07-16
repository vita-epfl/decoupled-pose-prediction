import torch
import numpy as np


class args():
    def __init__(self):
        self.dtype        = 'train'
        self.loader_workers = 1
        self.loader_shuffle = False
        self.pin_memory     = False
        self.device         = 'cuda'
        self.batch_size     = 100

args = args()
                                                  
import DataLoader_test


test = DataLoader_test.data_loader_test()

import DataLoader
train = DataLoader.data_loader(args)

