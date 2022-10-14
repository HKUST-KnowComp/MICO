import torch
from torch.nn import functional as F
import os
import sys
import random
import json
import numpy as np
from torch.utils.data import Dataset

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


def read_jsonl(input_file):

    records = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            temp_record = json.loads(line)
            guid = temp_record['qID']
            sentence = temp_record['sentence']
            opt1 = temp_record['option1']
            opt2 = temp_record['option2']
            label = temp_record['answer']

            conj = '_'
            idx = sentence.index(conj)
            context = sentence[:idx]
            option_str = '_' + sentence[idx + len(conj):].strip()
            option1 = option_str.replace('_', opt1)
            option2 = option_str.replace('_', opt2)

            if label == '1':
                records.append([sentence[:idx+len(conj)], option1, option2])
            elif label == '2':
                records.append([sentence[:idx+len(conj)], option2, option1])
            else:
                records.append([sentence[:idx+len(conj)], option1, option2])

    return records



class ConDataset(Dataset):

    def __init__(self, features):

        self.data = []

        for feature in features:
            self.data.append([feature[0], feature[1], feature[2]])
    
        self.length = len(self.data)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]
         

