from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import utils
import json
import tqdm, random
import argparse
import random
from vllm import LLM
from vllm import SamplingParams
from gensim_bm25 import bm25
import pprint,os
from retrieve_process import *
from gensim_vector import ReVector
random.seed(42)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, help="")
    parser.add_argument('--embedding_model', type=str, help="")
    parser.add_argument('--vector_path', type=str, help="", default="/home/zhoushiqi/workplace/apr/data/vectors/all_vector_2048.jsonl")
    parser.add_argument('--data_path', type=str, help="")
    parser.add_argument('--dest_path', type=str, help="")
    parser.add_argument('--model_id', type=str, help="")
    parser.add_argument('--codebase_path', type=str, help="")
    parser.add_argument('--gpu_index', type=int, default=0, help="")
    parser.add_argument('--num_gpus', type=int, default=1, help="")
    parser.add_argument('--num_per_iter', type=int, default=10, help="")
    parser.add_argument('--temperature', type=float, default=1, help="")
    parser.add_argument('--top_p', type=float, default=0.95, help="")
    parser.add_argument('--N', type=int, default=10, help="")
    parser.add_argument('--num_example', type=int, default=1, help="")
    parser.add_argument('--num_ticket', type=int, default=3, help="")
    parser.add_argument('--num_voter', type=int, default=10, help="")
    parser.add_argument('--uniform_weight', type=int, default=10, help="")
    # parser.add_argument('--max_len', type=int, default=512, help="")
    args = parser.parse_args()
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
dest_path = args.dest_path
all_error = 0
model_id = args.model_id

result_path = args.result_path
diff_datas = pre_process_diff(result_path, args.data_path, args.num_voter, uniform_weight=args.uniform_weight)
gap = len(diff_datas)//8
if args.gpu_index == 7:
    diff_datas = diff_datas[7*gap:]
else:
    diff_datas = diff_datas[args.gpu_index*gap:(args.gpu_index+1)*gap]

print(len(diff_datas))
embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
embedding_model = AutoModel.from_pretrained(args.embedding_model, trust_remote_code=True).to('cuda')
codebase = json.load(open(args.codebase_path))
vectors = utils.read_jsonl(args.vector_path)
vectors = {v['index']:v for v in vectors}
vec_models = {}
for k in codebase:
    vi = [{'id':d['id'], 'embedding':vectors[d['id']]['diff_embedding']} for d in codebase[k]]
    vec_model = ReVector(embedding_model, embedding_tokenizer, codebase[k], vi)
    vec_models[k] = vec_model
    
for diff in tqdm.tqdm(diff_datas):
    examples = vote_vec(vec_models[diff['rule_id']], diff['diffs'], weights=diff['weights'], k1=args.num_ticket, k2=args.num_example)
    diff['examples'] = examples

if os.path.exists(dest_path):
    results = utils.read_jsonl(dest_path)
    exsists = [r['id'] for r in results]
else:
    exsists = []  
    results = []
llm = LLM(seed=42, model=model_id, tensor_parallel_size=args.num_gpus,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
for i, data in tqdm.tqdm(enumerate(diff_datas)):
    if data['id'] in exsists:
        continue
    examples = data['examples']
    prompt = ""
    for j, example in enumerate(examples):
        prompt += "### Fix ESLint error in the following JavaScript code:\n"
        prompt += "### Buggy JavaScript\n"
        prompt += f"```\n{example['buggy_code']}\n```\n"
        # prompt += f"{example['buggy_code']}\n"
        prompt += f"\"rule_id\" : {example['rule_id']}\n"
        prompt += f"\"evidence\" : {example['evidence']}\n"
        prompt += f"\"warning_line\" : {example['warning_line']}\n"
        prompt += "### Fixed JavaScript\n"
        prompt += f"```\n{example['fixed_code']}\n```\n"
        # prompt += f"{example['fixed_code']}\n"
        prompt += "END_OF_DEMO\n"
    prompt += "### Fix ESLint error in the following JavaScript code:\n"
    prompt += "### Buggy JavaScript\n"
    prompt += f"```\n{data['buggy_code']}\n```\n"
    # prompt += f"{data['buggy_code']}\n"
    prompt += f"\"rule_id\" : {data['rule_id']}\n"
    prompt += f"\"evidence\" : {data['evidence']}\n"
    prompt += f"\"warning_line\" : {data['warning_line']}\n"
    prompt += "### Fixed JavaScript\n"
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    input_len = len(tokenizer.encode(prompt))
    if input_len > 15360:
        results.append({"id":data['id'],"prompt":prompt, "result":['']*args.N, "input_len":input_len})
        all_error += 1
        continue
    one_result = []
    # print(inputs)
    for i in range(0, args.N, args.num_per_iter):
        if i+args.num_per_iter > args.N:
            k = args.N - i
        else:
            k = args.num_per_iter
        sampling_params = SamplingParams(n=k, temperature=args.temperature, top_p=args.top_p, stop=[tokenizer.eos_token], min_tokens=0.8*len(tokenizer.encode(data['buggy_code'])), max_tokens=16384)#, stop_token_ids=[128009]
        completions = llm.generate(inputs, sampling_params)
        for output in [completions[0].outputs[i].text for i in range(k)]:
            output = output.split('[/INST]')[-1]        
            one_result.append(output)
        # print(one_result[-1])
    results.append({"id":data['id'],"prompt":prompt, "result":one_result, "input_len":input_len})
    utils.write_jsonl(dest_path, results)
print(all_error)

