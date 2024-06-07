from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import utils
import json
from transformers import AutoTokenizer
import numpy as np
from gensim.similarities.docsim import MatrixSimilarity
def softmax(x):
    x = np.array(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()
device = "cuda"
class ReVector:
    def __init__(self, model_id, codebase, vector_ids, index_path=None, save_path=None) -> None:
        self.id2document = {data['index']:data for data in codebase}
        self.vector_ids = vector_ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.doc_vectors = [[(i, x) for i, x in enumerate(vec['embedding'])] for vec in vector_ids]
        if index_path != None:
            self.index = MatrixSimilarity.load(index_path)
        else:
            self.index = MatrixSimilarity(self.doc_vectors, num_features=len(self.doc_vectors[0]))
            if save_path !=None:
                self.index.save(save_path)
        
    
    def process_text(self, s):
        s = s.strip()
        s = s.replace('\t', '    ')
        inputs = self.tokenizer.encode(s, return_tensors="pt", truncation=True, max_length=2048).to(device)
        embedding = self.model(inputs)[0].to("cpu").detach().numpy().tolist()
        return [(i, x) for i, x in enumerate(embedding)]
    
    
    def query(self, query_text, k):

        query_vec = self.process_text(query_text)

        # 计算与查询的相似度
        sims = self.index[query_vec]
        rank = sorted(list(range(len(self.doc_vectors))), key=lambda x:sims[x], reverse=True)
        results = []
        indexs = []
        for i in rank[:k]:
            data_index = self.vector_ids[i]['index']

            indexs.append(data_index)
            results.append(self.id2document[data_index])
        return results, indexs, softmax([sims[i] for i in rank[:k]])

if __name__ == '__main__':
    codebase = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/megadiff-single-function/process_filtered2048.jsonl")
    vector_ids = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/vectors/all_vector_2048.jsonl")
    vector_model = ReVector("/home/zhoushiqi/workplace/model/codet5p-110m-embedding", codebase, vector_ids)
    print(vector_model.query(codebase[1]["diff_context"], 1)[0][0]["buggy_function"] == codebase[1]["buggy_function"])
