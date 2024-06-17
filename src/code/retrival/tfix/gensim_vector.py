from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import utils
import json
from transformers import AutoTokenizer
import numpy as np
from gensim.similarities.docsim import MatrixSimilarity
from sentence_transformers import SentenceTransformer
def softmax(x):
    x = np.array(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()
device = "cuda"
class ReVector:
    def __init__(self, model, tokenizer, codebase, vector_ids, index_path=None, save_path=None) -> None:
        self.id2document = {data['id']:data for data in codebase}
        self.vector_ids = vector_ids
        self.tokenizer = tokenizer
        # self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.model = model
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
        # embedding = self.model.encode(s, convert_to_tensor=True).to("cpu").detach().numpy().tolist()
        inputs = self.tokenizer.encode(s, return_tensors="pt").to(device)
        if len(inputs[0]) > 512:
            inputs = inputs[:, :512]
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
            data_index = self.vector_ids[i]['id']

            indexs.append(data_index)
            results.append(self.id2document[data_index])
        return results, indexs, softmax([sims[i] for i in rank[:k]])

if __name__ == '__main__':
    codebase = json.load(open("/home/zhoushiqi/workplace/apr/data/tfix/data/train_data_clean_sm.json"))
    k = list(codebase.keys())[0]
    vector_ids = utils.read_jsonl("/home/zhoushiqi/workplace/apr/data/vectors/tfix_clean_sm_codet5p.jsonl")
    vector_ids ={v['index']:v for v in vector_ids }
    vi = [{'id':d['id'], 'embedding':vector_ids[d['id']]['diff_embedding']} for d in codebase[k]]

    embedding_tokenizer = AutoTokenizer.from_pretrained('/home/zhoushiqi/workplace/model/codet5p-110m-embedding', trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained('/home/zhoushiqi/workplace/model/codet5p-110m-embedding', trust_remote_code=True).to('cuda')
    vector_model = ReVector(embedding_model, embedding_tokenizer, codebase[k], vi)
    print(vector_model.query(codebase[k][0]["diff_context"], 10)[0][0]['buggy_code']==codebase[k][0]["buggy_code"])