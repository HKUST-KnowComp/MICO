
# dropout for roberta-large
# droupot is optional for roberta-base and bert-base

CUDA_VISIBLE_DEVICES=0 python main.py \
    --temp 0.07 \
    --save_folder ./ckpts_atomic/k2/roberta_large \
    --batch_size 196 \
    --max_seq_length 32 \
    --learning_rate 0.000005 \
    --epochs 10 \
    --save_freq 3 \
    --model roberta-large \
    --tokenizer_name roberta-large \
    --trainfile ../preprocess/ATOMIC-Ind-train.txt \
    --valfile ../preprocess/ATOMIC-Ind-valid.txt \
    --dropout \
    --k 2

