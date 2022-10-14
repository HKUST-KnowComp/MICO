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
from dataset_eval_commonsenseQA import *
from loss import EvalLossMulti


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


    parser.add_argument('--testfile', type=str, default='../CSQA/commonsenseQA/dev_rand_split.jsonl', \
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
    labels = []
    for idx, sents in enumerate(eval_loader):
        with torch.no_grad():
            data_time.update(time.time(), - end)

            features = []

            for sent in sents[:6]:
                ids = tokenizer(sent, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')

                if torch.cuda.is_available():
                    ids = {k: v.cuda() for k, v in ids.items()}

                embs = model(**ids)
                features.append(embs)

            option_features = torch.cat(features[1:], dim=0)
            output = criterion(features[0], option_features)
            label = sents[6].numpy()
            labels.append(label)

            batch_time.update(time.time() - end)
            end = time.time()

            outputs.append(output.topk(1, dim=-1)[1].squeeze().item())

    # calculate accuracy
    outputs = np.array(outputs)
    targets = np.concatenate(labels)
    acc1 = sum(np.equal(outputs, targets)) / outputs.shape[0]
    print(acc1)

    if 'test' in opt.testfile:
        fw = open(os.path.join(opt.save_folder, 'commonQA_test_pred.txt'), 'w')
    elif 'dev' in opt.testfile:
        fw = open(os.path.join(opt.save_folder, 'commonQA_dev_pred.txt'), 'w')

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
    pre_model = torch.load(os.path.join(opt.save_folder, 'best_model.pth'))
    print(pre_model.keys())
    model.load_state_dict(pre_model['model'])
    model = model.cuda()

    print('loading loss ...')
    criterion = EvalLossMulti(temperature=opt.temp)
    criterion = criterion.cuda()

    eval(test_loader, model, tokenizer, criterion, opt)

 
if __name__ == "__main__":
    main()
