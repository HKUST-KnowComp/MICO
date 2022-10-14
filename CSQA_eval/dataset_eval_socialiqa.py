import torch
from torch.nn import functional as F
import os
import sys
import random
import json
import numpy as np
import re
from torch.utils.data import Dataset

from util import load_tokenizer


def get_relation(question):

    if 'Why did' in question:
        rel = question.strip('?').replace('Why did', '')
    elif 'What will' in question:
        rel = question.strip('?').replace('What will', '')
    elif 'What does' in question:
        rel = question.strip('?').replace('What does', '')
    elif 'How would' in question:
        rel = question.strip('?').replace('How would', '')
    elif 'What would' in question:
        rel = question.strip('?').replace('What would', '')
    elif 'What did' in question:
        rel = question.strip('?').replace('What did', '')
    elif 'How will' in question:
        rel = question.strip('?').replace('How will', '')
    else:
        rel = None
    return rel


QUESTION_TO_ANSWER_PREFIX = {
    "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
    "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
    "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
    "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
    "What will (.*) do next?": r"[SUBJ] then",
    "How would (.*) feel after?": r"[SUBJ] then",
    "How would you describe (.*)?": r"[SUBJ] is seen as",
    "What kind of person is (.*)?": r"[SUBJ] is seen as",
    "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
    "Why did (.*) do that?": r"Before, [SUBJ] wanted",
    "Why did (.*) do this?": r"Before, [SUBJ] wanted",
    "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
    "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
    "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
    "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
    "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
    "What will happen to (.*)?": r"[SUBJ] then",
    "What will happen to (.*) next?": r"[SUBJ] then"
}


def get_relation_new(question):

    answer_prefix = ''
    for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
        m = re.match(template, question)
        if m is not None:
            answer_prefix = ans_prefix.replace('[SUBJ]', m.group(1).replace('?', ''))
            break

    print(m, answer_prefix)
    if answer_prefix == '':
        answer_prefix = question.replace('?', 'is')

    return answer_prefix


def read_jsonl(input_file, labelfile):

    records = []
    f = open(input_file, "r", encoding="utf-8-sig")
    fl = open(labelfile)
    for line, label in zip(f, fl):
        temp_record = json.loads(line)
        sentence = temp_record['context']
        name = sentence.split(' ')[0]
        rel = get_relation_new(temp_record['question'])
        sentence = sentence  + rel

        opt1 = temp_record['answerA']
        opt2 = temp_record['answerB']
        opt3 = temp_record['answerC']

        records.append([sentence, opt1, opt2, opt3, int(label)-1])

    return records

def read_test_jsonl(input_file):

    records = []
    f = open(input_file, "r", encoding="utf-8-sig")
    for line in f:
        temp_record = json.loads(line)
        sentence = temp_record['context']
        name = sentence.split(' ')[0]
        rel = get_relation_new(temp_record['question'])
        sentence = sentence  + rel

        opt1 = temp_record['answerA']
        opt2 = temp_record['answerB']
        opt3 = temp_record['answerC']
        label = 1

        records.append([sentence, opt1, opt2, opt3, int(label)-1])

    return records
 

class ConDataset(Dataset):

    def __init__(self, features):

        self.data = []

        for feature in features:
            self.data.append(feature)
    
        self.length = len(self.data)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        return self.data[idx]
         

    
