import json, os
from datasets import load_dataset, concatenate_datasets
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

def load_megadiff_dataset(dir_path):
    dataset_pool = []
    for file in os.listdir(dir_path):
        file_path = dir_path + f'/{file}'
        dataset = load_dataset('parquet', data_files=file_path)
        dataset_pool.append(dataset['train'])
    concatenated = concatenate_datasets(dataset_pool)
    return concatenated
    