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
###########################
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import torch
tokenizer = RobertaTokenizer.from_pretrained('/home/zhoushiqi/workplace/model/codet5_commit')
model = T5ForConditionalGeneration.from_pretrained('/home/zhoushiqi/workplace/model/codet5_commit')
import pickle
for dir in os.listdir("/home/zhoushiqi/workplace/apr/data/MCMD_raw/java"):
    print(dir)
    for file in os.listdir(f"/home/zhoushiqi/workplace/apr/data/MCMD_raw/java/{dir}"):
        file = f"/home/zhoushiqi/workplace/apr/data/MCMD_raw/java/{dir}/{file}"
        repo_raw_data = pickle.load(open(file,"rb"))
        i = 0
        for data in repo_raw_data:
            diff = data['diff']
            messege = data['msg']
            input_ids = tokenizer(diff, return_tensors="pt").input_ids
            if len(input_ids[0]) < 512:
                i += 1
    print(i, len(repo_raw_data))

# print(model(input_ids, decoder_input_ids=torch.tensor([[1]])).encoder_last_hidden_state.shape)
# this prints "{user.username}"
# generated_ids = model.generate(input_ids, max_length=100)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
###############################
