from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import utils
import tqdm, random
import argparse
import random
from vllm import LLM
from vllm import SamplingParams
import pprint,os, json
random.seed(42)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datas_path', type=str, help="")
    parser.add_argument('--dest_path', type=str, help="")
    parser.add_argument('--model_id', type=str, help="")
    parser.add_argument('--codebase_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--num_gpus', type=int, default=1, help="")
    parser.add_argument('--num_per_iter', type=int, default=10, help="")
    parser.add_argument('--temperature', type=float, default=1, help="")
    parser.add_argument('--top_p', type=float, default=0.95, help="")
    parser.add_argument('--N', type=int, default=10, help="")
    parser.add_argument('--num_example', type=int, default=10, help="")
    parser.add_argument('--num_gpu', type=int, help="")
    # parser.add_argument('--max_len', type=int, default=512, help="")
    args = parser.parse_args()
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
datas_path = args.datas_path
datas = json.load(open(datas_path))
gap = len(datas)//8
if args.num_gpu == 7:
    datas = datas[gap*7:]
else:
    datas = datas[args.num_gpu*gap:(args.num_gpu+1)*gap]
print(len(datas))
dest_path = args.dest_path
all_error = 0
model_id = args.model_id
llm = LLM(seed=42,model=model_id, tensor_parallel_size=args.num_gpus,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if os.path.exists(dest_path):
    results = utils.read_jsonl(dest_path)
    exsists = [r['id'] for r in results]
else:
    exsists = []
    results = []
codebase = json.load(open(args.codebase_path))
for k in codebase:
    print(f"{k}:{len(codebase[k])}")
for i, data in tqdm.tqdm(enumerate(datas)):
    if data['id'] in exsists:
        continue
    examples = random.sample(codebase[data['rule_id']], k=min(args.num_example, len(codebase[data['rule_id']])))
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
        # sampling_params = SamplingParams(use_beam_search=True, n=k, temperature=0, top_p=1, stop=[tokenizer.eos_token], min_tokens=0.8*len(tokenizer.encode(data['buggy_code'])), max_tokens=16384)#, stop_token_ids=[128009]
        completions = llm.generate(inputs, sampling_params)
        for output in [completions[0].outputs[i].text for i in range(k)]:
            output = output.split('[/INST]')[-1]        
            one_result.append(output)
    results.append({"id":data['id'],"prompt":prompt, "result":one_result, "input_len":input_len})
    utils.write_jsonl(dest_path, results)
print(all_error)
