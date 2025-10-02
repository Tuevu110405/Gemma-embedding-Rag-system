import re 
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.rag.vectorstore import VectorDB

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str):
        return self.extract_answer(text)
    
    def extract_answer(self,
                       text_response,
                       pattern : str = r"Answer:\s*(.*)"
                       ):
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text 
        else:
            return text_response
        
class Offline_RAG:
    def __init__(self,
                 llm,
                 vector_db : VectorDB,
                 
                 ) -> None:
        self.llm = llm
        self.vector_db = vector_db 
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()


    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_chain(self, top_k: int = 5, weights : list = [0.8, 0.2]):
        rag_chain = (
            {
                "context": lambda query: self.format_docs(self.vector_db.hybrid_search(query, top_k=top_k, weights=weights)),
                "question" : RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain