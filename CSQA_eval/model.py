import torch
import torch.nn as nn
import torch.nn.functional as F

from util import load_pretrained_model


class LModel(nn.Module):
    def __init__(self, encoder_name):
        super(LModel, self).__init__()

        self.encoder, hdim = load_pretrained_model(encoder_name)


    def forward(self, input_ids, attention_mask, token_type_ids=None):

        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True,
                              return_dict=True)

        last_hidden = output.last_hidden_state
        hidden_states = output.hidden_states
        pooler_output = output.pooler_output

        # cls token
        feat = last_hidden[:, 0, :]

        return feat
        
