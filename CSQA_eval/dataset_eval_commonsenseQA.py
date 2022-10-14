import torch
from torch.nn import functional as F
import os
import sys
import random
import json
import numpy as np
from torch.utils.data import Dataset

from util import load_tokenizer
import pandas as pd

def read_jsonl(input_file):

    records = []

    with open(input_file, 'r') as f:
        for line in f:
            temp_record = json.loads(line)
            question = temp_record['question']['stem']
            label = ['A', 'B', 'C', 'D', 'E'].index(temp_record['answerKey']) if "answerKey" in temp_record else None
            choices = [c['text'] for c in temp_record['question']['choices']]
            records.append((question, choices, label))

    return records


class ConDataset(Dataset):

    def __init__(self, features):

        self.data = []
        for feature in features:
            data_sent = []
            data_sent.append(feature[0])

            for sent in feature[1]:
                data_sent.append(sent)

            data_sent.append(feature[2])
            self.data.append(data_sent)
    
        self.length = len(self.data)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        return self.data[idx]
 
