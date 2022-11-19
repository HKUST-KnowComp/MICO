from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
from torch.utils.data import DataLoader

from util import load_tokenizer, AverageMeter, accuracy

from model import LModel
from loss_cs import EvalLoss
from dataset import *
import pickle


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--model', type=str, default='bert-base')
    parser.add_argument('--config_name', type=str, default=None, help='pretrained config name')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max context tokens length') 
    parser.add_argument('--tokenizer_name', type=str, default='bert-base', help='tokenizer name')
    parser.add_argument('--save_folder', type=str, default='./ckpts', help='path to checkpoints')

    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--testfile', type=str, default='../preprocess/ATOMIC-Ind-train.txt', \
                        help='test file path')

    opt = parser.parse_args()

    return opt


def eval(eval_loader, model, tokenizer, criterion, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    outputs_head = []
    outputs = []
    for idx, (anchor, pos) in enumerate(eval_loader):
        if idx % 100 == 0:
            print('eval batch: ', idx)

        with torch.no_grad():
            data_time.update(time.time(), - end)

            features = []
            anchor_ids = tokenizer(anchor, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
            pos_ids = tokenizer(pos[0], padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')

            if torch.cuda.is_available():
                anchor_ids = {k: v.cuda() for k, v in anchor_ids.items()}
                pos_ids = {k: v.cuda() for k, v in pos_ids.items()}

            anchor_emb = model(**anchor_ids)
            pos_emb = model(**pos_ids)

            if torch.cuda.is_available():
                anchor_emb = anchor_emb.cpu()
                pos_emb = pos_emb.cpu()

            outputs_head.append(anchor_emb)
            outputs.append(pos_emb)

    # tails contains the entities
    outputs = np.concatenate(outputs, axis=0)
    # heads for test
    outputs_head = np.concatenate(outputs_head, axis=0)

    assert 'CN' in opt.testfile or 'ATOMIC' in opt.testfile, 'Unknown dataset'

    if 'CN' in opt.testfile:
        dataset_name = 'CN'
    else 'ATOMIC' in opt.testfile:
        dataset_name = 'ATOMIC'

    if 'train' in opt.testfile:
        with open(os.path.join(opt.save_folder, '{}_tails_train.pkl'.format(dataset_name)), 'wb') as f:
            pickle.dump(outputs, f)
    elif 'valid' in opt.testfile:
        with open(os.path.join(opt.save_folder, '{}_tails_valid.pkl'.format(dataset_name)), 'wb') as f:
            pickle.dump(outputs, f)
    elif 'test' in opt.testfile:
        with open(os.path.join(opt.save_folder, '{}_tails_test.pkl'.format(dataset_name)), 'wb') as f:
            pickle.dump(outputs, f)

        with open(os.path.join(opt.save_folder, '{}_heads_test.pkl'.format(dataset_name)), 'wb') as f:
            pickle.dump(outputs_head, f)


def main():

    opt = parse_option()

    print('loading data ...')
    tokenizer = load_tokenizer(opt.tokenizer_name)
    raw_test_data = read_kg(opt.testfile)

    test_dataset = ConDataset(raw_test_data)

    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print('loading model ...')

    model = LModel(opt.model)
    pre_model = torch.load(os.path.join(opt.save_folder, 'best_model.pth'))
    print(pre_model.keys())
    model.load_state_dict(pre_model['model'])
    model = model.cuda()

    print('loading loss ...')
    criterion = EvalLoss(temperature=opt.temp)
    criterion = criterion.cuda()

    eval(test_loader, model, tokenizer, criterion, opt)

 
if __name__ == "__main__":
    main()
