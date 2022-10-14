alias python=~/tools/anaconda3py37/bin/python

CUDA_VISIBLE_DEVICES=0 python transform_copa.py \
  --lm gpt2-large \
  --database_file ../CSQA_eval/dataset/COPA/copa-dev-new.jsonl \
  --out_dir ./results_gpt2_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_copa.py \
  --lm gpt2-large \
  --database_file ../CSQA_eval/dataset/COPA/copa-test-new.jsonl \
  --out_dir ./results_gpt2_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_socialiqa.py \
  --lm gpt2-large \
  --database_file ../CSQA_eval/dataset/SIQA/dev.jsonl \
  --label_file ../CSQA_eval/dataset/SIQA/dev-labels.lst \
  --out_dir ./results_gpt2_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_commonQA.py \
  --lm gpt2-large \
  --database_file ../CSQA_eval/dataset/CSQA/dev_rand_split.jsonl \
  --out_dir ./results_gpt2_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_copa.py \
  --lm roberta-large \
  --database_file ../CSQA_eval/dataset/COPA/copa-dev-new.jsonl \
  --out_dir ./results_roberta_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_copa.py \
  --lm roberta-large \
  --database_file ../CSQA_eval/dataset/COPA/copa-test-new.jsonl \
  --out_dir ./results_roberta_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_socialiqa.py \
  --lm roberta-large \
  --database_file ../CSQA_eval/dataset/SIQA/dev.jsonl \
  --label_file ../CSQA_eval/dataset/SIQA/dev-labels.lst \
  --out_dir ./results_roberta_large \
  --device 0

CUDA_VISIBLE_DEVICES=0 python transform_commonQA.py \
  --lm roberta-large \
  --database_file ../CSQA_eval/dataset/CSQA/dev_rand_split.jsonl \
  --out_dir ./results_roberta_large \
  --device 0

