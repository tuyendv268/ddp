import numpy as np
import pandas as pd
import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from utils import norm_text

class BM25:                    
    def train(self, texts):
        self.texts = texts
        
        texts['bm25_text'] = texts["passage_text"].apply(lambda x: norm_text(x))
        corpus = [x.split() for x in texts['bm25_text'].values]
                
        self.dictionary = Dictionary(corpus)
        bm25_model = OkapiBM25Model(dictionary=self.dictionary)
        bm25_corpus = bm25_model[list(map(self.dictionary.doc2bow, corpus))]

        self.bm25_index = SparseMatrixSimilarity(
            bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
            normalize_queries=False, normalize_documents=False)

        self.tfidf_model = TfidfModel(dictionary=self.dictionary, smartirs='bnn')
        
        
    def get_topk(self, query, topk: int=10):
        query = norm_text(query)
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query.split())]
        scores = self.bm25_index[tfidf_query]
        
        if topk == None:
            top_n = np.argsort(scores)[::-1]
        else:
            top_n = np.argsort(scores)[::-1][:topk]
            
        return top_n, scores[top_n]
    
    def ranking(self, query, top_k: int=100):
        query = norm_text(query)
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query.split())]
        scores = self.bm25_index[tfidf_query]
        
        top_n = np.argsort(scores)[::-1][:top_k]
        
        result = []
        for rank, index in enumerate(top_n):
            _id = self.texts.id.values[index]
            _score = scores[index]
            _rank = rank
            _text = self.texts.passage_text.values[index]
            
            result.append([_id, _score, _rank, _text])
            
        result = pd.DataFrame(result, columns=["id", "retrieval_score", "retrieval_rank", "retrieval_text"])
        return result
        

if __name__ == "__main__":
    from utils import norm_question, norm_text
    bm25_model = BM25("outputs/bm25")
    query = "mô tả về tổng công ty mobifone"
    
    question = norm_question(query)
    tokenized_query = norm_text(query)
    top_n, bm25_scores  = bm25_model.get_text_topk(tokenized_query, topk=5)

    for t in top_n:
        print(t, end="\n\n")
    print(bm25_scores)