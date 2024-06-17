from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import utils
import os
import argparse
import pprint
import tqdm
import torch
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dest_path', type=str, help="")
    parser.add_argument('--model_id', type=str, help="")
    parser.add_argument('--codebase_path', type=str, help="")
    parser.add_argument('--gpu_index', type=int, default=0, help="")

    args = parser.parse_args()
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

checkpoint = args.model_id
device = "cuda"  # for GPU usage or "cpu" for CPU usage
kvs = json.load(open(args.codebase_path))
datasets = []
for k in kvs:
    datasets += kvs[k]
gap = len(datasets)//8 + 1
datasets = datasets[args.gpu_index*gap:(args.gpu_index+1)*gap]

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
# model = SentenceTransformer(checkpoint).to(device)
# tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
# model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
vecs = []
err = 0
for data in tqdm.tqdm(datasets):
    inputs_diff = tokenizer.encode(data['diff_context'], return_tensors="pt").to(device)
    inputs_bug = tokenizer.encode(data['buggy_code'], return_tensors="pt").to(device)
    # input_ids = tokenizer(data['diff_context'], return_tensors="pt").input_ids.to(device)
    if len(inputs_diff) > 512 or len(inputs_bug) > 512:
        err += 1
        continue
    embedding_diff = model(inputs_diff)[0]
    embedding_bug = model(inputs_bug)[0]
    # embedding_diff = model.encode(data['diff_context'], convert_to_tensor=True)
    # embedding_bug = model.encode(data['buggy_code'], convert_to_tensor=True)
    # embedding = torch.mean(model(input_ids, decoder_input_ids=torch.tensor([[1]]).to(device)).encoder_last_hidden_state[0], dim=0, keepdim=False)
    vecs.append({"diff_embedding":embedding_diff.to("cpu").detach().numpy().tolist(), "bug_embedding":embedding_bug.to("cpu").detach().numpy().tolist(),"index":data['id']})
utils.write_jsonl(args.dest_path, vecs)
print(err)