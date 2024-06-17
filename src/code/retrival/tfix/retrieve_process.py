import utils
import tqdm
import json
from gensim_bm25 import bm25
from gensim_vector import ReVector
from transformers import AutoTokenizer
from collections import defaultdict
import re
import math
    
def pre_process_diff(result_path, data_path, num_voter, uniform_weight=True):
    plausible_ids = []
    x = json.load(open(result_path))
    id2weight = {}
    for id in x:
        results = x[id][:num_voter]
        if "passed" in results:
            plausible_ids.append(id)
            continue
        else:
            weights = []
            for result in results:
                if result == "":
                    weights.append(0)
                else:
                    weights.append(1)
            id2weight[id] = weights
    buggy = json.load(open(data_path))
    diff_datas = []
    for bug in buggy:
        if str(bug['id']) in plausible_ids or str(bug['id']) not in x:
            continue
        bug['diffs'] = x[str(bug['id'])][:num_voter]

        if uniform_weight:
            bug['weights'] = [1] * num_voter
        else:
            bug['weights'] = id2weight[str(bug['id'])]
        diff_datas.append(bug)
    return diff_datas
    


def vote(bm25_model, qs, weights, k1=3, k2=1):
    records = defaultdict(int)
    for q, w in zip(qs, weights):
        _, indexs, scores = bm25_model.query(q, k1)
        for index, score in zip(indexs, scores):
            records[index] += w*score
    rank = sorted(list(records.keys()), key=lambda x:records[x], reverse=True)
    return rank[:k2]
def vote_vec(vec_model, qs, weights, k1=3, k2=1):
    records = defaultdict(int)
    for q, w in zip(qs, weights):
        results, indexs, scores = vec_model.query(q, k1)
        # print("-----------")
        # print(results)
        for index, score in zip(indexs, scores):
            records[index] += w*score
    rank = sorted(list(records.keys()), key=lambda x:records[x], reverse=True)
    return [vec_model.id2document[j] for j in rank[:k2]]