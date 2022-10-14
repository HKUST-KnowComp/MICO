from __future__ import print_function

import os
import sys
import argparse
import time
import math
import logging

import torch

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_tokenizer

from loss import CSLoss
from model import LModel
from dataset import *

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW, get_constant_schedule_with_warmup


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='nnumber of training epochs')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='bert-base')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom path')
    parser.add_argument('--save_folder', type=str, default='./ckpts', help='path to checkpoints')
    parser.add_argument('--temp', type=float, default=0.7,
                        help='temperature for loss function')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base', help='tokenizer name')
    parser.add_argument('--trainfile', type=str, default='../preprocess/ATOMIC-Ind-train.txt',\
                        help='model name or path')
    parser.add_argument('--valfile',  type=str, default='../preprocess/ATOMIC-Ind-valid.txt')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max context tokens length')
    parser.add_argument('--k', type=int, default=2, help='candidate positive tails during training')
    parser.add_argument('--dropout', action='store_true', help='use dropout for hidden feature')

    opt = parser.parse_args()

    return opt


def train(train_loader, model, tokenizer, criterion, optimizer, epoch, logger, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (anchor, p) in enumerate(train_loader):
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        anchor_ids = tokenizer(anchor, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
        pos_tail_ids = []
        for i in range(len(p)):
            p_ids = tokenizer(p[i], padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
            pos_tail_ids.append(p_ids)

        if torch.cuda.is_available():
            anchor_ids = {k: v.cuda() for k, v in anchor_ids.items()}
            pos_tail_ids = [{k: v.cuda() for k, v in p_ids.items()} for p_ids in pos_tail_ids]

        anchor_emb = model(**anchor_ids)
        pos_tail_emb = [model(**p_ids) for p_ids in pos_tail_ids]

        bsz = anchor_ids['input_ids'].shape[0]

        loss = criterion(anchor_emb, pos_tail_emb)
        losses.update(loss.item(), bsz)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]['lr']

        if (idx + 1) % opt.print_freq == 0:
           print('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'lr {lr:.7f}\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
              epoch, idx+1, len(train_loader), batch_time=batch_time,
              data_time=data_time, lr=lr, loss=losses))
           logger.info('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'lr {lr:.7f}\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
              epoch, idx+1, len(train_loader), batch_time=batch_time,
              data_time=data_time, lr=lr, loss=losses))

           sys.stdout.flush()

    return losses.avg

def eval(val_loader, model, tokenizer, criterion, optimizer, epoch, logger, opt):
    model.eval()

    losses = AverageMeter()
    with torch.no_grad():
        for idx, (anchor, p) in enumerate(val_loader):
            anchor_ids = tokenizer(anchor, padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
            p_ids = []
            for i in range(len(p)):
                p_id = tokenizer(p[i], padding=True, truncation=True, max_length=opt.max_seq_length, return_tensors='pt')
                p_ids.append(p_id)

            if torch.cuda.is_available():
                anchor_ids = {k: v.cuda() for k, v in anchor_ids.items()}
                p_ids = [{k:v.cuda() for k, v in p_id.items()} for p_id in p_ids]

            anchor_emb = model(**anchor_ids)
            p_embs = [model(**p_id) for p_id in p_ids]

            bsz = anchor_ids['input_ids'].shape[0]

            loss = criterion(anchor_emb, p_embs)
            losses.update(loss.item(), bsz)

    logger.info('Eval: [{0}][{1}]\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
              epoch, len(val_loader),
              loss=losses))
    sys.stdout.flush()

    return losses.avg

def main():

    opt = parse_option()

    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    np.random.seed(42)
    random.seed(42)

    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    logging.basicConfig(filename='{}/train.log'.format(opt.save_folder),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    print('loading training data ...')
    logger.info('loading training data ...')
    tokenizer = load_tokenizer(opt.tokenizer_name)
    raw_train_data = read_kg(opt.trainfile)
    print(len(raw_train_data))

    train_dataset = ConDataset(raw_train_data, opt.k)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    raw_valid_data = read_kg(opt.valfile)
    valid_dataset = ConDataset(raw_valid_data)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    total_steps = int(len(train_dataset) / opt.batch_size * opt.epochs)
    
    print('loading model ...')
    logger.info('loading model ...')
    print('use dropout: ', opt.dropout)
    model = LModel(opt.model, opt.dropout)
    model = model.cuda()

    print('loading loss ...')
    criterion = CSLoss(temperature=opt.temp)
    criterion = criterion.cuda()

    optimizer = AdamW(model.parameters(), lr=opt.learning_rate)
    best_loss = 100.0
    current_loss = 100.0

    fw = open(os.path.join(opt.save_folder, 'val_loss.log'), 'w')
    for epoch in range(1, opt.epochs + 1):
        time1 = time.time()
        loss = train(train_loader, model, tokenizer, criterion, optimizer, epoch, logger, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        valid_loss = eval(valid_loader, model, tokenizer, criterion, optimizer, epoch, logger, opt)

        print('Epoch: {}\t valid loss: {:.3f}\n'.format(epoch, valid_loss))
        fw.write('Epoch: {}\t valid loss: {:.3f}\n'.format(epoch, valid_loss))
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_file = os.path.join(opt.save_folder, 'best_model.pth')
            save_model(model, optimizer, opt, opt.epochs, save_file)

        # early stop
        if abs(valid_loss - current_loss) / current_loss < 0.01:
            logger.info('early stop at epoch: {}'.format(epoch))
            break

        current_loss = valid_loss

    fw.close() 

if __name__ == "__main__":
    main()
