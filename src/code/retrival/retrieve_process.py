import utils
import tqdm
from gensim_bm25 import bm25
from gensim_vector import ReVector
from transformers import AutoTokenizer
from collections import defaultdict
import re
import math
    
def pre_process_diff(result_path, info_path, process_path, num_voter, context_window=3, uniform_weight=True):
    plausible_ids = []
    x = utils.read_jsonl(result_path)
    id2weight = {}
    for xx in x:
        results = xx['result'][:num_voter]
        if 'other test Failing tests: 0\n' in results:
            plausible_ids.append(f"{xx['project']}_{xx['bug_id']}")
        else:
            weights = []
            avg = []
            for i,r in enumerate(results):
                if "other" in r:
                    weights.append(1.5)
                if "Repetite Failed":
                    weights.append(0)
                if r == "Compile Failing" or r == "trigger test" or r == "Time out":
                    weights.append(0.5)
                if "trigger testFailing" in r:
                    weights.append(1)
                # else:
                #     n = re.findall("Failing tests: (\d+)\n", r)
                #     assert len(n) == 1, r
                #     n = n[0]
                #     if n==1:
                #         weights.append(1)
                #     else:
                #         weights.append(1.5)
                #     avg.append(int(n))
            # if avg == []:
            #     bias = 0
            # else:
            #     bias = (-1*math.log(sum(avg)/len(avg)*1.5)+20)/20
            # weights = [w if w!='x' else bias for w in weights]
            id2weight[f"{xx['project']}_{xx['bug_id']}"] = weights
    buggy = utils.read_jsonl(info_path)
    answer = utils.read_jsonl(process_path)[0]
    diff_datas = []
    for bug in buggy:
        new_data = {}
        bug_id = f"{bug['project']}_{bug['bug_id']}"
        if bug_id in plausible_ids or bug_id not in answer:
            continue
        buggy_code = bug['erro_repairs'][0]['src_code'][0].strip()
        new_data['project'] = bug['project']
        new_data['bug_id'] = bug['bug_id']
        new_data['buggy_code'] = buggy_code
        new_data['fixed'] = [a.strip() for a in answer[bug_id]][:num_voter]
        new_data['diffs'] = [utils._process({'buggy_function':buggy_code, 'fixed_function':a}, context_window) for a in new_data['fixed']]
        if uniform_weight:
            new_data['weights'] = [1] * len(new_data['fixed'])
        else:
            new_data['weights'] = id2weight[bug_id]
        new_data['weights'] = [0 if diff==buggy_code else new_data['weights'][i] for i, diff in enumerate(new_data['fixed'])]
        # print(new_data['weights'])
        diff_datas.append(new_data)
    return diff_datas
    
#bm25 codebase
def build_model(codebase_path, model_name="/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct"):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    codebase = utils.read_jsonl(codebase_path)
    diffs = [code['diff_context'] for code in codebase]
    tokenized_diffs = [code['tokenized_diff_context'] for code in codebase]
    bm25_model = bm25(tokenizer, diffs, tokenized_diffs)
    return bm25_model, codebase
#vector codebase
def build_model_vec(codebase_path, model_name="/home/zhoushiqi/workplace/model/codet5p-110m-embedding", vecter_path=""):
    # 加载tokenizer
    codebase = utils.read_jsonl(codebase_path)
    vector_ids = utils.read_jsonl(vecter_path)
    vec_model = ReVector(model_name, codebase, vector_ids, save_path="/home/zhoushiqi/workplace/apr/data/vectors/index_2048")
    return vec_model, codebase
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
        _, indexs, scores = vec_model.query(q, k1)
        for index, score in zip(indexs, scores):
            records[index] += w*score
    rank = sorted(list(records.keys()), key=lambda x:records[x], reverse=True)
    return [vec_model.id2document[j] for j in rank[:k2]]