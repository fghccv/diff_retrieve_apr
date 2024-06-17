import json
import re,utils,difflib,tqdm
from collections import defaultdict
erro = 0
import argparse, os, utils
f = open("/home/zhoushiqi/workplace/apr/src/code/retrival/tfix/temp.txt", 'w')
def em_metric(data_path, dest_path):
    results = []
    ground_truth = json.load(open("/home/zhoushiqi/workplace/apr/data/tfix/data/test_data_clean_sm.json"))
    ground_truth = {x['id']:x for x in ground_truth}
    process_datas = json.load(open(data_path))
    records = defaultdict(list)
    for data in tqdm.tqdm(process_datas):
        true_fix = ground_truth[data['id']]['fixed_code']
        bug = ground_truth[data['id']]['buggy_code']
        p = 0
        for r in data['result'][:args.num]:
            # f.write('r\n')
            # f.write(r.strip().replace('\n', '').replace('\t', '').replace(' ', '')+'\n')
            # f.write('true\n')
            # f.write(true_fix.strip().replace('\n', '').replace('\t', '').replace(' ', '')+'\n')
            # f.write('============================\n')
            if r.strip().replace('\n', '').replace('\t', '').replace(' ', '') == true_fix.strip().replace('\n', '').replace('\t', '').replace(' ', ''):
                p = 1
                records[data['id']].append("passed")
            else:
                records[data['id']].append(utils._process({'buggy_function':bug.strip(), 'fixed_function':r.strip()}, 3))
                
        results.append(p)
    if args.write:
        json.dump(records, open(dest_path, 'w'))
    return sum(results),len(results), sum(results)/len(results)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, help="")
    parser.add_argument('--dest_path', type=str, help="")
    parser.add_argument('--metric', type=str, help="")
    parser.add_argument('--write', type=int, help="", default=1)
    parser.add_argument('--num', type=int, help="", default=100)
    args = parser.parse_args()
    if args.metric == 'em':
        print(f"ac: {em_metric(args.data_path, args.dest_path)}")
