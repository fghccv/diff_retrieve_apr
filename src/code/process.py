import json
import re,utils
erro = 0
import argparse, os, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help="")
    parser.add_argument('--dest_path', type=str, help="")
    args = parser.parse_args()
def process_one_sample(sample):
    global erro
    if "### refined code" in sample.lower():
        index = sample.lower().index("### refined code")
        sample = sample[index:]
    its = re.findall("```java((.|\n)*?)```", sample)
    if its == []:
        its = re.findall("```((.|\n)*?)```", sample)
    if its != []:
        return its[0][0]
    else:
        erro += 1
        return sample

data_path = args.data_path
dest_path = args.dest_path
datas = utils.read_jsonl(data_path)
new_datas = {}
for data in datas:
    id = f"{data['project']}_{data['bug_id']}"
    result = [process_one_sample(sample) for sample in data['result']]
    new_datas[id] = result
print(erro)
json.dump(new_datas, open(dest_path, 'w'))
    