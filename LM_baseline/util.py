import torch
from transformers import GPT2Tokenizer, RobertaTokenizer, BertTokenizer, GPT2LMHeadModel, RobertaForMaskedLM, BertForMaskedLM

def init_model(model_name:str, device:torch.device):
    if model_name == 'gpt2-large':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif model_name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        print('check if model name in (gpt2-large, roberta-large, roberta-base, bert-base)')

    if model_name == 'gpt2-large':
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name == 'roberta-large':
        model = RobertaForMaskedLM.from_pretrained(model_name)
    elif model_name == 'roberta-base':
        model = RobertaForMaskedLM.from_pretrained(model_name)
    elif model_name == 'bert-base':
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    model.to(device)
    model.eval()
    return model, tokenizer

