from __future__ import print_function

import torch
import torch.nn as nn


class CSLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(CSLoss, self).__init__()
        self.temp = temperature
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, f1, fs):

        if len(fs) == 1:
            cos_sim = self.cos(f1.unsqueeze(1), fs[0].unsqueeze(0)) / self.temp
            labels = torch.arange(cos_sim.size(0))
            labels = labels.to(cos_sim.device)
            loss = self.loss_fct(cos_sim, labels)
 
        else:
            # (bs, bs)
            cos_scores = []
            for f_pos in fs:
                cos_score = self.cos(f1.unsqueeze(1), f_pos.unsqueeze(0)) /self.temp
                cos_scores.append(cos_score)

            mask = torch.eye(cos_scores[0].size(0))
            mask_neg = 1 - mask
            mask = mask.to(cos_scores[0].device)
            mask_neg = mask_neg.to(cos_scores[0].device)

            cos_scores_neg = [mask_neg*cos_score for cos_score in cos_scores]
            # (bs, bs*k), all the negatives
            cos_new = torch.cat(cos_scores_neg, 1)

            # (bs, bs), except diagonal, 1000. > 1.0/temp
            max_pos = torch.ones(cos_scores[0].shape) * 1000.
            max_pos = max_pos.to(cos_scores[0].device)
            max_pos = max_pos * mask_neg

            # (bs, bs*k)
            cos_pos_blank = [max_pos for i in range(len(fs))]
            cos_pos_blank = torch.cat(cos_pos_blank, 1)
            # (bs, bs*k), only diagonal
            cos_pos_new_tmp = torch.cat([mask * cos_score for cos_score in cos_scores], 1)
            # make values 1000.0 except the diagonal score
            cos_pos_new = cos_pos_new_tmp + cos_pos_blank
            cos_min = torch.min(cos_pos_new, dim=1, keepdims=True)[1]

            # the min score lies on the original diagonal
            labels = cos_min.squeeze(1)

            #(bs, bs*k)
            mask_pos_final = torch.zeros(cos_pos_new.shape)
            mask_pos_final = mask_pos_final.to(cos_scores[0].device)
            mask_pos_final.scatter_(1, cos_min, 1.0, reduce='add')

            #(bs, bs*k), negative+positive
            cos_sim = cos_new + cos_pos_new * mask_pos_final
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
