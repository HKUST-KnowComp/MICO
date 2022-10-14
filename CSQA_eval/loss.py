from __future__ import print_function

import torch
import torch.nn as nn


class CSLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(CSLoss, self).__init__()
        self.temp = temperature
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.hard_negative_weight = 1.0

    def forward(self, f1, f2):
        # (bs, bs)
        cos_sim = self.cos(f1.unsqueeze(1), f2.unsqueeze(0)) /self.temp

        labels = torch.arange(cos_sim.size(0))
        labels = labels.to(cos_sim.device)

        loss = self.loss_fct(cos_sim, labels)

        return loss


class EvalLoss(nn.Module):

    def __init__(self, temperature=0.7):
        super(EvalLoss, self).__init__()

        self.temp = temperature
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.hard_negative_weight = 1.0


    def forward(self, f1, f2, f3):

        cos_sim = self.cos(f1, f2) / self.temp
        cos_sim_neg = self.cos(f1, f3) / self.temp

        labels = torch.tensor([0]*f1.size(0))
        labels = labels.to(cos_sim.device)

        cos_sim = torch.cat([cos_sim, cos_sim_neg], 0)
        cos_sim = cos_sim.unsqueeze(0)

        loss = self.loss_fct(cos_sim, labels)
        return loss, cos_sim


class EvalLossMulti(nn.Module):

    def __init__(self, temperature=0.07):
        super(EvalLossMulti, self).__init__()

        self.temp = temperature
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, f1, features):

        cos_sim = self.cos(f1, features) / self.temp
        return cos_sim 

