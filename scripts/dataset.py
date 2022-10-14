import torch
from torch.nn import functional as F
import os
import sys
import random
import json
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

from util import load_tokenizer

def read_kg(input_file):

    records = []
    with open(input_file, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            if len(info) != 5:
                continue
            anchor, tail, label, ind, _ = info
            records.append([anchor, tail, label, ind])
    return records


class ConDataset(Dataset):

    def __init__(self, features, k=1):

        self.data = []
        self.tails = defaultdict(list)
        self.k = k

        for feature in features:
            self.data.append([feature[0], feature[1], feature[2]])
            # label, positive (same anchor)
            self.tails[feature[2]].append(feature[1])
    
        self.length = len(self.data)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        label = self.data[idx][2]
        pos_tails = [self.data[idx][1]]
        for i in range(self.k-1):
            rand_ind = np.random.randint(0, len(self.tails[label]))
            another_pos = self.tails[label][rand_ind]
            pos_tails.append(another_pos)

        return self.data[idx][0], pos_tails
         
  
