import numpy as np
import os, sys
import argparse
import json
import torch
import re
from util import init_model

from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

def get_relation(question):

    if 'Why did' in question:
        rel = question.strip('?').replace('Why did', '')
    elif 'What will' in question:
        rel = question.strip('?').replace('What will', '')
    elif 'What does' in question:
        rel = question.strip('?').replace('What does', '')
    elif 'How would' in question:
        rel = question.strip('?').replace('How would', '')
    elif 'What would' in question:
        rel = question.strip('?').replace('What would', '')
    elif 'What did' in question:
        rel = question.strip('?').replace('What did', '')
    elif 'How will' in question:
        rel = question.strip('?').replace('How will', '')
    else:
        rel = None
    return rel

QUESTION_TO_ANSWER_PREFIX = {
    "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
    "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
    "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
    "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
    "What will (.*) do next?": r"[SUBJ] then",
    "How would (.*) feel after?": r"[SUBJ] then",
    "How would you describe (.*)?": r"[SUBJ] is seen as",
    "What kind of person is (.*)?": r"[SUBJ] is seen as",
    "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
    "Why did (.*) do that?": r"Before, [SUBJ] wanted",
    "Why did (.*) do this?": r"Before, [SUBJ] wanted",
    "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
    "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
    "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
    "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
    "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
    "What will happen to (.*)?": r"[SUBJ] then",
    "What will happen to (.*) next?": r"[SUBJ] then"
}


def get_relation_new(question):

    answer_prefix = ''
    for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
        m = re.match(template, question)
        if m is not None:
            answer_prefix = ans_prefix.replace('[SUBJ]', m.group(1).replace('?', ''))
            break

    if answer_prefix == '':
        answer_prefix = question.replace('?', 'is')

    return answer_prefix



def read_jsonl(input_file, labelfile):

    records = []
    f = open(input_file, "r", encoding="utf-8-sig")
    fl = open(labelfile)
    for line, label in zip(f, fl):
        temp_record = json.loads(line)
        sentence = temp_record['context']
        name = sentence.split(' ')[0]
        rel = get_relation_new(temp_record['question'])

        sentence = sentence  + rel
        opt1 = sentence + ' ' + temp_record['answerA']
        opt2 = sentence + ' ' + temp_record['answerB']
        opt3 = sentence + ' ' + temp_record['answerC']

        records.append([sentence, opt1, opt2, opt3, int(label)-1])

    return records
 

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def logits_score(logits, context_tokens, opt_tokens):
    score = 0.0

    #(1, len1)
    end_index = opt_tokens.shape[1] - 1
    #(1, len2)
    start_index = context_tokens.shape[1] - 1
    for i in range(end_index - start_index):
        m = softmax(logits[0][start_index+i])
        score += np.log(m[opt_tokens[0][start_index+i+1]])

    score = score/(end_index-start_index+1)

    return score


def reader(record):

    context, opt1, opt2, opt3, label = record
    opt1 = context + ' ' + opt1
    opt2 = context + ' ' + opt2
    opt3 = context + ' ' + opt3
    return context, opt1, opt2, opt3, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--database_file", default=None, type=str, required=True, help="jsonl file for evaluation")
    parser.add_argument("--label_file", default=None, type=str, required=True, help='label file for evaluation')
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device)

    set_name = 'socialiqa'
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_file = os.path.join(args.out_dir, f"{args.lm}_{set_name}_predictions.txt")
    gold = []
    predictions = []

    with torch.no_grad():
        f_out = open(out_file, 'w')
        records = read_jsonl(args.database_file, args.label_file)
        for i, record in enumerate(records):
            context_text, opt1, opt2, opt3, label = reader(record)
            gold.append(label)

            opt_token1 = tokenizer(opt1, return_tensors='pt')
            opt_token2 = tokenizer(opt2, return_tensors='pt')
            opt_token3 = tokenizer(opt3, return_tensors='pt')

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

            pred = np.argmin(scores)
            predictions.append(pred)
            f_out.write('{}\n'.format(pred))

        if None not in gold:
            accuracy = accuracy_score(gold, predictions)
            print(f'Accuracy: {accuracy:.3f}')

 

if __name__ == "__main__":

    main()
