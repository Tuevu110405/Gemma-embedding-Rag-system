from typing import Union
import torch
# from langchain_chroma import Chroma
# from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from src.rag.bm25_util import normalize_score_bm25, tokenize_bm25
from langchain_core.documents import Document

class VectorDB:
    def __init__(
            self,
            documents = None,
            
            # vector_db : Union[Chroma, FAISS] = FAISS
            # embedding = HuggingFaceEmbeddings()
    ):
        # self.vector_db = vector_db 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300M").to(device=self.device)
        self.documents = documents
        self.semantic_index = self._build_db(self.documents)
        self.bm25, self.bm25_id = self._build_rv(self.documents)
        
    def _build_db(self, documents):
        # create embeddings gamma from documents 
        embeddings = []
        with torch.no_grad():
            
            embedding = self.embedding_model.encode(
                    documents,
                    convert_to_tensor=True,
                    show_progress_bar = True
                )
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embeddings.append(embedding.cpu().numpy())

        #initialize faiss semantic index
        embeddings = np.vstack(embeddings).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print("Initialize semantic index successfully")
        return index
    
    def _build_rv(self, documents):
        #initialize bm25 index 
        tokenized_docs = [tokenize_bm25(doc) for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs)
        bm25_id = []
        for i in range(len(tokenized_docs)):
            bm25_id.append(i)
        
        return bm25_index, bm25_id



    # def _build_db(self, documents):
    #     db = self.vector_db.from_documents(
    #         documents=documents,
    #         embedding = self.embedding
    #     )

    #Hybrid search for retriever
    def semantic_search(self, query , top_k: int):
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor = True
            ).to(self.device)
        with torch.no_grad():
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)

        distance, index = self.semantic_index.search(query_embedding.astype("float32"), top_k)


        return distance[0], index[0]
    
    def text_search(self, query, top_k: int):
        tokenized_query = tokenize_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        normalized_scores = normalize_score_bm25(scores)
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        results = [(self.bm25_id[idx], normalized_scores[idx]) for idx in top_indices]
        return results 
    
    def hybrid_search(self, query, top_k: int, weights: list = [0.5, 0.5]):
        # weights[0] is for semantic  search, weights[1] is for textual search
        sem_distance, sem_index = self.semantic_search(query, top_k)
        text_results = self.text_search(query, top_k)

        combined_scores = {}
        for idx, score in zip(sem_index, sem_distance):
            combined_scores[idx] = combined_scores.get(idx, 0) + weights[0] * score

        for idx, score in text_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + weights[1] * score

        # save content of docs in the results
        
        sorted_results = sorted(list(combined_scores.items()), key=lambda x: x[1], reverse=True)[:top_k]
        final_docs = []
        for idx, score in sorted_results:
            # Create a Document object for each result
            doc = Document(
                page_content=self.documents[idx], 
                metadata={"id": idx, "score": score}
            )
            final_docs.append(doc)
            
        return final_docs