from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import utils
import os
import argparse
import pprint
import tqdm
import torch
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
datasets = utils.read_jsonl(args.codebase_path)
gap = len(datasets)//8 + 1
datasets = datasets[args.gpu_index*gap:(args.gpu_index+1)*gap]

# tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
# model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
# model = SentenceTransformer(checkpoint).to(device)
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
vecs = []
err = 0
for data in tqdm.tqdm(datasets):
    # inputs = tokenizer.encode(data['diff_context'], return_tensors="pt").to(device)
    input_ids = tokenizer(data['diff_context'], return_tensors="pt").input_ids.to(device)
    if len(input_ids[0]) > 512:
        err += 1
        continue
    # embedding = model(inputs)[0]
    # embedding = model.encode(data['diff_context'], convert_to_tensor=True)
    embedding = torch.mean(model(input_ids, decoder_input_ids=torch.tensor([[1]]).to(device)).encoder_last_hidden_state[0], dim=0, keepdim=False)
    vecs.append({"embedding":embedding.to("cpu").detach().numpy().tolist(), "index":data['index']})
utils.write_jsonl(args.dest_path, vecs)
print(err)