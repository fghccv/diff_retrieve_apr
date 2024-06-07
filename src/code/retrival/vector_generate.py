from transformers import AutoModel, AutoTokenizer
import utils
import os
import argparse
import pprint
import tqdm
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

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
vecs = []
err = 0
for data in tqdm.tqdm(datasets):
    inputs = tokenizer.encode(data['diff_context'], return_tensors="pt").to(device)
    if len(inputs[0]) > 512:
        continue
    embedding = model(inputs)[0]
    vecs.append({"embedding":embedding.to("cpu").detach().numpy().tolist(), "index":data['index']})
utils.write_jsonl(args.dest_path, vecs)
print(err)