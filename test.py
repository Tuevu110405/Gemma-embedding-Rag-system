from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG 
from src.base.llm_model import get_hf_llm

def build_rag_chain(llm, data_dir, data_type, weights: list = [0.8, 0.2], top_k : int = 5):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers= 2)
    retriever = VectorDB(documents = doc_loaded)
    rag_chain = Offline_RAG(llm, retriever).get_chain(top_k = top_k, weights = weights)
    return rag_chain

if __name__ == '__main__':

    llm = get_hf_llm()
    data_dir = 'data_source'
    data_type = 'pdf'
    rag_chain = build_rag_chain(llm = llm, data_dir=data_dir, data_type=data_type)
    # doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers= 2)
    # retriever = VectorDB(documents = doc_loaded)
    question = 'What is attetion transformer'
    result = rag_chain.invoke(question)
    print(result)



