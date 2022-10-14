from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
#from transformers import *
from transformers import RobertaModel, BertModel, RobertaTokenizer, BertTokenizer


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(traget.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.num_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + match.cos(math.pi * epoch / args.epochs)) / 2

    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)

        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, save_file)

    del state


def load_pretrained_model(name):
    if name == 'roberta-base':
        model = RobertaModel.from_pretrained('roberta-base')
        hdim = 768
    elif name == 'roberta-large':
        model = RobertaModel.from_pretrained('roberta-large')
        hdim = 1024
    elif name == 'bert-large':
        model = BertModel.from_pretrained('bert-large-uncased')
        hdim = 1024
    else: #bert-base
        model = BertModel.from_pretrained('bert-base-uncased')
        hdim = 768
    return model, hdim


def load_tokenizer(name):
    if name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    elif name == 'bert-large':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    else: #bert-base
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

