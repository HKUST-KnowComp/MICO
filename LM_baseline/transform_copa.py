import os
import re
import json
import tqdm
import torch
import logging
import argparse
import numpy as np

from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as etree
from util import init_model


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def logits_score(logits, context_tokens, skeleton_tokens):
    score = 0.0
    start_index = context_tokens.shape[1] - 1
    end_index = skeleton_tokens.shape[1] - 1

    for i in range(end_index - start_index):
        m = softmax(logits[0][start_index+i])
        score += np.log(m[skeleton_tokens[0][start_index+i+1]])
    score = score/(end_index-start_index+1)

    return score


def reader(fields):

    context = fields['sentence']
    label = fields['answer']
    choices = [fields['option1'], fields['option2']]

    label = int(label) - 1
    raw_text1 = context.replace("_", '')
    raw_text2 = context.replace("_", '')

    idx = context.index("_")
    skeleton1 = context[:idx] + choices[0]
    skeleton2 = context[:idx] + choices[1]

    return raw_text1, raw_text2, skeleton1, skeleton2, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--database_file", default=None, type=str, required=True, help="jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device)

    set_name = os.path.basename(args.database_file).replace(".jsonl", "")
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_file = os.path.join(args.out_dir, f"{args.lm}_{set_name}_predictions.txt")
    gold = []
    predictions = []

    loss_fct = CrossEntropyLoss()

    with torch.no_grad():
        f_out = open(out_file, "w")

        with open(args.database_file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                raw_text1, raw_text2, skeleton1, skeleton2, label = reader(fields)
                gold.append(label)

                skeleton_token1 = tokenizer(skeleton1, return_tensors='pt')
                skeleton_token2 = tokenizer(skeleton2, return_tensors='pt')

                outputs1 = model(input_ids=skeleton_token1['input_ids'].to(device),
                               attention_mask=skeleton_token1['attention_mask'].to(device),
                               labels=skeleton_token1['input_ids'].to(device))
                score1 = outputs1[0]

                outputs2 = model(input_ids=skeleton_token2['input_ids'].to(device),
                               attention_mask=skeleton_token2['attention_mask'].to(device),
                               labels=skeleton_token2['input_ids'].to(device))

                score2 = outputs2[0]

                if score1 < score2:
                    prediction = 0
                else:
                    prediction = 1

                predictions.append(prediction)

                f_out.write(json.dumps(fields) + '\n')

    if None not in gold:
        accuracy = accuracy_score(gold, predictions)
        print(f'Accuracy: {accuracy:.3f}')


if __name__ == "__main__":
    main()

    
