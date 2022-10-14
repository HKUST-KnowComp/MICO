# MICO
This is the code repo for EMNLP2022 MICO: multiview contrastive learning framework for commonsense knowledge representation

## Data Preparation

### Training Data
   Download [ATOMIC19 and CN-82k](https://github.com/BinWang28/InductivE) and put the `dataset_only` folder under `./preprocess`
   Prepare the training data

   ```
   cd ./preprocess
   python mapping_train_name.py
   ```    

### Evaluation Data
   ```
   cd ./CSQA_eval
   ```
   [COPA](https://people.ict.usc.edu/~gordon/copa.html)
   Transform the original data into the form sastifying the format of MICO.
   ```
   cd ./datasets/copa/
   python transform.py
   ```
   [CommmonsenseQA (CSQA)](https://allenai.org/data/commonsenseqa) and 
   [SocialIQA (SIQA)](https://leaderboard.allenai.org/socialiqa/submissions/get-started)
     

## Training

   ```
   cd ./scripts

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
   ```


## Evaluation for Zero-shot Commonsense Question Answering

   Use pre-trained LMs to evaluate the CSQA tasks.
   ```
   cd ./LM_baseline
   sh eval_baseline.sh 
   ```
   It will report the accuracy score on the three tasks and generate the prediction file.


   Use MICO to evaluate the CSQA tasks. For example, use checkpoint trained on ATOMIC to test SIQA
   ```
   cd ../CSQA_eval

   CUDA_VISIBLE_DEVICES=0 python evaluate_socialiqa.py --save_folder ../scripts/ckpts_atomic/k2/roberta_large \
       --max_seq_length 64 \
       --temp 0.07 \
       --model roberta-large \
       --tokenizer_name roberta-large \
       --testfile ../dataset/SIQA/socialiqa-train-dev/dev.jsonl \
       --testlabel ../dataset/SIQA/socialiqa-train-dev/dev-labels.lst
   ```
   More evaluation refers to `eval.sh`.

## Evaluation for Inductive CSKG Completion

   Use trained models to generate feature first and then retrieve, calculate the MRR and rank top10 score.

   First extract feature of training dataset and test dataset. This step will generate pickle files for CSKG completion
   ```
   cd ./scripts
   sh eval_tail.sh
   ```

   Then for ATOMIC19
   ```
   python eval_mrr_atomic.py
   ```
   and for ConceptNet
   ```
   python eval_mrr_cn.py
   ```

