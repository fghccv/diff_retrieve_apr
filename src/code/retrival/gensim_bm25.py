from gensim import corpora
from gensim.summarization.bm25 import BM25
from tqdm import tqdm
import utils
import json
from transformers import AutoTokenizer
import numpy as np

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()

class bm25:
    def __init__(self, tokenizer, document, tokenized_document=None) -> None:
        self.tokenizer = tokenizer
        self.document = document
        self.texts = tokenized_document if tokenized_document != None\
                else self.process_document()
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.bm25 = BM25(self.corpus)
        total_docs = len(self.corpus)
        self.average_idf = sum(float(val) for val in self.bm25.idf.values()) / total_docs

    def process_text(self, s):
        s = s.strip()
        s = s.replace('\t', '    ')
        s = self.tokenizer.tokenize(s)
        return s
    
    def process_document(self):
        texts = []
        for doc in self.document:
            doc = self.process_text(doc)
            doc = self.tokenizer.tokenize(doc)
            texts.append(doc)
        return texts
    
    def query(self, query_text, k):

        query_tokens = self.process_text(query_text)
        query_bow = self.dictionary.doc2bow(query_tokens)

        # 计算与查询的相似度
        scores = self.bm25.get_scores(query_bow, self.average_idf)
        rank = sorted(list(range(len(self.corpus))), key=lambda x:scores[x], reverse=True)
        results = [self.document[i] for i in rank[:k]]
        return results, rank[:k], softmax([scores[i] for i in rank[:k]])
