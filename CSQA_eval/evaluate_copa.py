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
from dataset_eval_copa import *
from loss import EvalLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--model', type=str, default='bert-base')
    parser.add_argument('--config_name', type=str, default=None, help='pretrained config name')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max context tokens length') 
    parser.add_argument('--tokenizer_name', type=str, default='bert-base', help='tokenizer name')
    parser.add_argument('--save_folder', type=str, default='./ckpts', help='path to checkpoints')

    parser.add_argument('--temp', type=float, default=0.7,
                        help='temperature for loss function')


    parser.add_argument('--testfile', type=str, default='../dataset/COPA/copa-dev-new.jsonl', \
                        help='test file path')

    opt = parser.parse_args()

    return opt


def eval(eval_loader, model, tokenizer, criterion, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    outputs = []
    for idx, (anchor, c1, c2) in enumerate(eval_loader):
        with torch.no_grad():
            data_time.update(time.time(), - end)

            anchor_ids = tokenizer(anchor, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
            c1_ids = tokenizer(c1, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
            c2_ids = tokenizer(c2, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')

            if torch.cuda.is_available():
                anchor_ids = {k: v.cuda() for k, v in anchor_ids.items()}
                c1_ids = {k: v.cuda() for k, v in c1_ids.items()}
                c2_ids = {k: v.cuda() for k, v in c2_ids.items()}

            anchor_emb = model(**anchor_ids)
            c1_emb = model(**c1_ids)
            c2_emb = model(**c2_ids)

            loss, output = criterion(anchor_emb, c1_emb, c2_emb)
            losses.update(loss)
            batch_time.update(time.time() - end)
            end = time.time()

            outputs.append(output[0].topk(1, dim=-1)[1].squeeze().item())

    # calculate accuracy

    outputs = np.array(outputs)
    targets = np.zeros(outputs.shape[0])

    acc1 = sum(np.equal(outputs, targets)) / outputs.shape[0]
    print(acc1)

    if 'test' in opt.testfile:
        fw = open(os.path.join(opt.save_folder, 'COPA_test_pred.txt'), 'w')
    elif 'dev' in opt.testfile:
        fw = open(os.path.join(opt.save_folder, 'COPA_dev_pred.txt'), 'w')

    for pred in outputs:
        fw.write('{}\n'.format(pred+1))
    fw.close()


    return losses.avg, acc1


def main():

    opt = parse_option()

    print('loading data ...')
    tokenizer = load_tokenizer(opt.tokenizer_name)
    raw_test_data = read_jsonl(opt.testfile)
    test_dataset = ConDataset(raw_test_data)

    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print('loading model ...')

    model = LModel(opt.model)
    pre_model = torch.load(os.path.join(opt.save_folder, 'ckpt_epoch_6.pth'))
    model.load_state_dict(pre_model['model'])
    model = model.cuda()

    print('loading loss ...')
    criterion = EvalLoss(temperature=opt.temp)
    criterion = criterion.cuda()

    eval(test_loader, model, tokenizer, criterion, opt)

 
if __name__ == "__main__":
    main()
