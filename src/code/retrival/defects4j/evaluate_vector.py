from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import utils
import tqdm, random
import argparse
import random
from vllm import LLM
from vllm import SamplingParams
from gensim_vector import ReVector
import pprint,os
random.seed(42)
def q2f(q, k=1):
    results, indexs, _ = vector_model.query(q, k)
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datas_path', type=str, help="")
    parser.add_argument('--vector_path', type=str, help="")
    parser.add_argument('--embedding_model_path', type=str, help="")
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
    # parser.add_argument('--max_len', type=int, default=512, help="")
    args = parser.parse_args()
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
datas_path = args.datas_path
datas = utils.read_jsonl(datas_path)[args.start_index:args.end_index]
dest_path = args.dest_path
all_error = 0
model_id = args.model_id
llm = LLM(seed=42, model=model_id, tensor_parallel_size=args.num_gpus,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
codebase = utils.read_jsonl(args.codebase_path)
vector_ids = utils.read_jsonl(args.vector_path)
document = [code['buggy_function'] for code in codebase]
vector_model = ReVector(args.embedding_model_path, codebase, vector_ids)
if os.path.exists(dest_path):
    results = utils.read_jsonl(dest_path)
    exsists = [f"{r['project']}_{r['bug_id']}" for r in results]
else:
    exsists = []
    results = []
print('retrive')
buggycodes = [data['erro_repairs'][0]['src_code'][0] for data in datas]
all_examples = [q2f(bug, args.num_example) for bug in buggycodes]
for i, data in tqdm.tqdm(enumerate(datas)):
    if f"{data['project']}_{data['bug_id']}" in exsists:
        continue
    local_dir_path = data['local_dir_path']
    erro_func = data['test_func']
    erro_repair = data['erro_repairs'][0]
    buggy_code = erro_repair['src_code'][0]
    repair_code = erro_repair['fixs'][0]
    document = erro_repair['docs'][0][0]
    with open(local_dir_path+'/erro.java') as f:
        erro_messege = f.read()
    erro_messege = erro_messege.split('\n')[1]
    prompt = "As an debugger, you should refine the buggy program.\n"
    examples = all_examples[i]
    for j, example in enumerate(examples):
        prompt += f"### Example {j+1}\n"
        b, f = example['buggy_function'], example['fixed_function']
        b = b.replace('\t', '    ')
        f = f.replace('\t', '    ')
        prompt += f"### Buggy code:\n```java\n{b}\n```\n"
        prompt += f"### Refined code:\n```java\n{f}\n```\n"
    prompt += f"### Example {len(examples)+1}\n"
    prompt += f"### Program document:\n```text\n{document}\n```\n"
    prompt += f"### Failed test:\n```java\n{erro_func}\n```\n"
    prompt += f"### Test info:\n```text\n{erro_messege}\n```\n"
    prompt += f"### Buggy code:\n```java\n{buggy_code}\n```\n"
    prompt += "### Refined code:\n"
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    input_len = len(tokenizer.encode(prompt))
    if input_len > 10240:
        results.append({"project":data['project'],'bug_id':data['bug_id'],"prompt":prompt, "result":['']*args.N})
        all_error += 1
        continue
    one_result = []
    # print(inputs)
    for i in range(0, args.N, args.num_per_iter):
        if i+args.num_per_iter > args.N:
            k = args.N - i
        else:
            k = args.num_per_iter
        sampling_params = SamplingParams(n=k, temperature=args.temperature, top_p=args.top_p, max_tokens=int(1.2*input_len), min_tokens=int(0.8*len(tokenizer.encode(buggy_code))), stop=[tokenizer.eos_token])#, stop_token_ids=[128009]
        completions = llm.generate(inputs, sampling_params)
        for output in [completions[0].outputs[i].text for i in range(k)]:
            output = output.split('[/INST]')[-1]        
            one_result.append(output)
        # print(one_result[-1])
    results.append({"project":data['project'],'bug_id':data['bug_id'],"prompt":prompt, "result":one_result})
    utils.write_jsonl(dest_path, results)
print(all_error)
