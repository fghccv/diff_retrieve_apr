from rank_bm25 import BM25Okapi
from tqdm import tqdm
import utils
from transformers import AutoTokenizer
model_name = "/home/zhoushiqi/workplace/model/deepseek-coder-6.7b-instruct"  # 你可以换成任何Hugging Face模型名称

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

datasets = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process.jsonl")
corpus = [data['diff_context'] for data in datasets]
tokenize_corpus = [data['tokenize_diff_context'] for data in datasets]


bm25 = BM25Okapi(tokenize_corpus)

buggy = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/df4_process_data/one_function/1.2.jsonl")
answer = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/evaluate_results/deepseek/baseline_1.2_N100_T1_process.jsonl")[0]
b1 = buggy[0]
a1 = answer[b1['project'] + '_' + str(b1['bug_id'])][0]
data = {'buggy_function':b1['erro_repairs'][0]['src_code'][0], 'fixed_function':a1}
query = utils._process(data, 3)
tokenized_query = tokenizer.tokenize(query)
bm25.get_top_n(tokenized_query, corpus, n=1)