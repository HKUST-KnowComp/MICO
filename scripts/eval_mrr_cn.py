import json
import numpy as np
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

path_dir = 'ckpts_conceptnet/k2/roberta_large'

with open('./{}/all_heads_test.pkl'.format(path_dir), 'rb') as f:
    arr_heads_test = pickle.load(f)

with open('./{}/all_tails_test.pkl'.format(path_dir), 'rb') as f:
    arr_tails_test = pickle.load(f)

with open('./{}/all_tails_train.pkl'.format(path_dir), 'rb') as f:
    arr_tails_train = pickle.load(f)

with open('./{}/all_tails_valid.pkl'.format(path_dir), 'rb') as f:
    arr_tails_valid = pickle.load(f)

dicts = defaultdict(list)
ALL = []
ALL_train_test = []
for line in open('../preprocess/CN82k-Ind-test.txt'):
    info = line.strip().split('\t')
    ALL.append(info)
    ALL_train_test.append(info)
    dicts[info[0]].append(info[1])

for line in open('../preprocess/CN82k-Ind-train.txt'):
    info = line.strip().split('\t')
    ALL_train_test.append(info)
    dicts[info[0]].append(info[1])

for line in open('../preprocess/CN82k-Ind-valid.txt'):
    info = line.strip().split('\t')
    ALL_train_test.append(info)
    dicts[info[0]].append(info[1])

#same tail events
same_tails = defaultdict(list)
for i, info in enumerate(ALL_train_test):
    same_tails[info[1]].append(i)
#filter tail events inds
arr_all_tails = np.concatenate([arr_tails_test, arr_tails_train, arr_tails_valid], axis=0)
print(arr_all_tails.shape)

arr_unique_tails = []
arr_unique_labels = {}

for i, key in enumerate(same_tails.keys()):
    #tail text, tail embedding, label
    arr_unique_tails.append(arr_all_tails[same_tails[key][0]].reshape(1, -1))
    arr_unique_labels[key] = i

test_labels = []
for ind, info in enumerate(ALL):
    test_labels.append(arr_unique_labels[info[1]])

dicts_filter = {}
for key in dicts.keys():
    dicts_filter[key] = [arr_unique_labels[val] for val in dicts[key]]

print('test instance number: ', len(ALL))

arr_tail = np.concatenate(arr_unique_tails, axis=0)
print(arr_tail.shape)
print('all unique tail number: ', arr_tail.shape[0])


hits = []
ranks = []
for i in range(10):
    hits.append([])

for ind_row in range(0, arr_heads_test.shape[0], 5000):
    arr_heads_test_split = arr_heads_test[ind_row:min(ind_row+5000, arr_heads_test.shape[0]), :]
    scores = cosine_similarity(arr_heads_test_split, arr_tail)

    for ind, (info, score) in enumerate(zip(ALL[ind_row:min(ind_row+5000, arr_heads_test.shape[0])], scores)):

        head = info[0]
        filter1 = dicts_filter[info[0]]

        target_value = score[test_labels[ind_row+ind]]

        score[filter1] = 0.0
        score[test_labels[ind_row+ind]] = target_value

        argsort1 = np.argsort(-score)
        rank1 = np.nonzero(argsort1 == test_labels[ind_row+ind])
        ranks.append(rank1[0][0]+1)

        for hit_level in range(0, 10):
            if rank1[0][0] <= hit_level:
                hits[hit_level].append(1.0)
            else:
                hits[hit_level].append(0.0)

for k in range(0, 10):
    print('Hits @{}: {}'.format(k+1, np.mean(hits[k])))

print('Mean rank: {}'.format(np.mean(ranks)))
print('Mrr: {}'.format(np.mean(1./np.array(ranks))))


