import os, sys
import numpy as np
import pandas as pd
import torch
import json
import argparse
from util import init_model

from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score


def read_jsonl(input_file):

    records = []

    with open(input_file, 'r') as f:
        for line in f:
            temp_record = json.loads(line)
            question = temp_record['question']['stem']
            label = ['A', 'B', 'C', 'D', 'E'].index(temp_record['answerKey']) if "answerKey" in temp_record else None
            choices = [c['text'] for c in temp_record['question']['choices']]
            records.append((question, choices, label))

    return records


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def logits_score(logits, context_tokens, skeleton_tokens):
    score = 0.0
    #(1, len1)
    end_index = skeleton_tokens.shape[1] - 1
    #(1, len2)
    start_index = context_tokens.shape[1] - 1
    for i in range(end_index - start_index):
        m = softmax(logits[0][start_index+i])
        score += np.log(m[skeleton_tokens[0][start_index+i+1]])

    score = score/(end_index-start_index+1)

    return score


def reader(record):

    context, opt1, opt2, opt3, label = record
    skeleton1 = context + ' ' + opt1
    skeleton2 = context + ' ' + opt2
    skeleton3 = context + ' ' + opt3
    return context, skeleton1, skeleton2, skeleton3, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--database_file", default=None, type=str, required=False, help="jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device)

    set_name = 'commonsenseQA'
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_file = os.path.join(args.out_dir, f"{args.lm}_{set_name}_predictions2.jsonl")
    gold = []
    predictions = []

    with torch.no_grad():
        f_out = open(out_file, 'w')
        records = read_jsonl(args.database_file)
        for i, record in enumerate(records):
            context_text, hypos, label = record
            gold.append(int(label))

            opt1 = context_text + ' ' + hypos[0]
            opt2 = context_text + ' ' + hypos[1]
            opt3 = context_text + ' ' + hypos[2]
            opt4 = context_text + ' ' + hypos[3]
            opt5 = context_text + ' ' + hypos[4]

            opt_token1 = tokenizer(opt1, return_tensors='pt')
            opt_token2 = tokenizer(opt2, return_tensors='pt')
            opt_token3 = tokenizer(opt3, return_tensors='pt')
            opt_token4 = tokenizer(opt4, return_tensors='pt')
            opt_token5 = tokenizer(opt5, return_tensors='pt')

            scores = []
            score1 = model(input_ids=opt_token1['input_ids'].to(device),
                               attention_mask=opt_token1['attention_mask'].to(device),
                               labels=opt_token1['input_ids'].to(device))[0]
            scores.append(score1.cpu().item())

            score2 = model(input_ids=opt_token2['input_ids'].to(device),
                               attention_mask=opt_token2['attention_mask'].to(device),
                               labels=opt_token2['input_ids'].to(device))[0]
            scores.append(score2.cpu().item())

            score3 = model(input_ids=opt_token3['input_ids'].to(device),
                               attention_mask=opt_token3['attention_mask'].to(device),
                               labels=opt_token3['input_ids'].to(device))[0]
            scores.append(score3.cpu().item())

            score4 = model(input_ids=opt_token4['input_ids'].to(device),
                               attention_mask=opt_token4['attention_mask'].to(device),
                               labels=opt_token4['input_ids'].to(device))[0]
            scores.append(score4.cpu().item())

            score5 = model(input_ids=opt_token5['input_ids'].to(device),
                               attention_mask=opt_token5['attention_mask'].to(device),
                               labels=opt_token5['input_ids'].to(device))[0]
            scores.append(score5.cpu().item())

            pred = np.argmin(scores)
            predictions.append(pred)
            f_out.write('{}\n'.format(pred))


        if None not in gold:
            accuracy = accuracy_score(gold, predictions)
            print(f'Accuracy: {accuracy:.3f}')


if __name__ == "__main__":

    main()
