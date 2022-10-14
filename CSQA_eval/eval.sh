CUDA_VISIBLE_DEVICES=0 python evaluate_socialiqa.py --save_folder ../scripts/ckpts_atomic19/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/SIQA/socialiqa-train-dev/dev.jsonl \
    --testlabel ./dataset/SIQA/socialiqa-train-dev/dev-labels.lst

CUDA_VISIBLE_DEVICES=0 python evaluate_socialiqa.py --save_folder ../scripts/ckpts_conceptnet/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/SIQA/socialiqa-train-dev/dev.jsonl \
    --testlabel ./dataset/SIQA/socialiqa-train-dev/dev-labels.lst 

CUDA_VISIBLE_DEVICES=0 python evaluate_commonsenseqa.py --save_folder ../scripts/ckpts_atomic19/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/CSQA/commonsenseQA/dev_rand_split.jsonl

CUDA_VISIBLE_DEVICES=0 python evaluate_commonsenseqa.py --save_folder ../scripts/ckpts_conceptnet/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/CSQA/commonsenseQA/dev_rand_split.jsonl

CUDA_VISIBLE_DEVICES=0 python evaluate_copa.py --save_folder ../scripts/ckpts_atomic19/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/COPA/copa-dev-new.jsonl

CUDA_VISIBLE_DEVICES=0 python evaluate_copa.py --save_folder ../scripts/ckpts_conceptnet/k2/roberta_large \
    --max_seq_length 64 \
    --temp 0.07 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --testfile ./dataset/COPA/copa-dev-new.jsonl


