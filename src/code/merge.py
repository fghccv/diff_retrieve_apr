
import argparse, os, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--merge_dir', type=str, default='/home/zhoushiqi/workplace/apr/data/df4_process_data/one_function/1.2.jsonl', help="")
    parser.add_argument('--dest_path', type=str, default="/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/baseline_1.2.json", help="")
    
    # parser.add_argument('--max_len', type=int, default=512, help="")
    args = parser.parse_args()
    all_datas = []
    for file in os.listdir(args.merge_dir):
        all_datas += utils.read_jsonl(args.merge_dir+'/{}'.format(file))
    print(len(all_datas))
    utils.write_jsonl(args.dest_path, all_datas)