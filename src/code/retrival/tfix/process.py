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
    sample = sample.split("### Fixed JavaScript\n")[-1]
    # index = sample.index("END_OF_DEMO") if "END_OF_DEMO" in sample else len(sample)
    # return sample[:index]
    its = re.findall("```((.|\n)*?)```", sample)
    if its == []:
        its = re.findall("javascript```((.|\n)*?)```", sample)
    if its != []:
        return its[0][0]
    else:
        erro += 1
        # print(sample[:50])
        # print('====================')
        return sample

data_path = args.data_path
dest_path = args.dest_path
datas = utils.read_jsonl(data_path)
new_datas = []
for data in datas:
    result = [process_one_sample(sample) for sample in data['result']]
    new_datas.append({'id':data['id'], 'result':result})
print(erro)
json.dump(new_datas, open(dest_path, 'w'))
    