import json, os, difflib, tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
def load_megadiff_dataset(dir_path):
    dataset_pool = []
    for file in os.listdir(dir_path):
        file_path = dir_path + f'/{file}'
        dataset = load_dataset('parquet', data_files=file_path)
        dataset_pool.append(dataset['train'])
    concatenated = concatenate_datasets(dataset_pool)
    return concatenated

def _process(data, context_window):
    assert 'buggy_function' in data and 'fixed_function' in data
    diff = list(difflib.unified_diff(data['buggy_function'].splitlines(), data['fixed_function'].splitlines(), n=context_window))
    # Remove the first two lines (the file headers)
    if diff and diff[0].startswith('---') and diff[1].startswith('+++'):
        diff = diff[2:]
    diff = '\n'.join(diff)
    return diff
def process(datasets, context_window=3):
    model_name = "/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct"  # 你可以换成任何Hugging Face模型名称

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_datasets = []
    for data in tqdm.tqdm(datasets):
        data['diff_context'] = _process(data, context_window)
        data['tokenize_diff_context'] = tokenizer.tokenize(data['diff_context'])
        new_datasets.append(data)
    return new_datasets
def read_jsonl(file_path):
    try:
        with open(file_path) as f:
            datas = []
            for line in f:
                line = json.loads(line)
                datas.append(line)
    except:
        with open(file_path, encoding="iso-8859-1") as f:
            datas = []
            for line in f:
                line = json.loads(line)
                datas.append(line)
    return datas

def write_jsonl(file_path, datas):
    with open(file_path, 'w') as f:
        for data in datas:
            f.write(json.dumps(data) + '\n')
def _split_bug_fix(diff):
    bug = []
    fix = []
    for l in diff.split('\n'):
        if l.startswith('+'):
            fix.append(l[1:])
        elif l.startswith('-'):
            bug.append(l[1:])
        else:
            fix.append(l[1:])
            bug.append(l[1:])
    return '\n'.join(bug), '\n'.join(fix)
def split_bug_fix(process_file_path, dest_path):
    process = read_jsonl(process_file_path)
    for data in tqdm.tqdm(process):
        bug, fix = _split_bug_fix(data['diff_context'])
        data['fl_bug_fix'] = {'bug':bug, 'fix':fix}
    write_jsonl(dest_path, process)
    
        
if __name__ == '__main__':
    # datasets = load_megadiff_dataset('/home/zhoushiqi/workplace/apr/data/megadiff-single-function/data')
    # process_datasets = process(datasets, 3)
    # write_jsonl("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process.jsonl", process_datasets)
    split_bug_fix("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048.jsonl", "/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048_add_fl_bfp.jsonl")