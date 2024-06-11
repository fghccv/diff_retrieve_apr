# from transformers import AutoModel, AutoTokenizer
# import utils
# from tqdm import tqdm
# from collections import Counter
# model_name = "/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct"  # 你可以换成任何Hugging Face模型名称

# # 加载tokenizerfrom gensim_bm25 import bm25
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# datasets = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process.jsonl")
# ls = []
# new_datasets = []
# index = 0
# err = 0
# for data in tqdm(datasets):
#     b = data["buggy_function"]
#     b = b.strip()
#     b = b.replace('\t', '    ')
#     tokenized_buggy_code = tokenizer.tokenize(b)
#     ls.append(len(tokenized_buggy_code))
#     data['index'] = index
#     index += 1
#     new_datasets.append(data)
# print(Counter(ls))
# # print(err)
# # utils.write_jsonl("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048.jsonl", new_datasets)
from utils import read_jsonl
x = read_jsonl("/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/retrieval/baseline_1.2_buggy_vector_k1_add_report_N200_T1_result.jsonl")
t = 0
for xx in x:
    if "other test Failing tests: 0\n" in xx['result'][:100]:
        t += 1
print(t)