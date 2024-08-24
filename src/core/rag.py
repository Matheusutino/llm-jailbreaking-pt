import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class RAG:
    def __init__(self, 
                 embedding_model_name: str = "text-embedding-3-small", 
                 index_filename: str = "vector_store/questions_index.pkl",
                 api_key: str = None):
        """
        Initializes the RAG class.

        Args:
            embedding_model_name (str): Name of the embeddings model (default: "text-embedding-3-small").
            index_filename (str): Filename to load the vector index (default: "questions_index.pkl").
        """
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name, api_key=self.api_key)
        self.index_filename = index_filename
        self.vectorstore = None
        self._load_index()

    def _load_index(self):
        """
        Loads the vector index from a file.
        """
        if os.path.exists(self.index_filename):
            try:
                self.vectorstore = Chroma(persist_directory=self.index_filename, embedding_function=self.embedding_model)
                print("Vector index loaded successfully.")
            except Exception as e:
                print(f"Error loading vector index: {e}")
                raise
        else:
            raise FileNotFoundError("Vector index file does not exist.")

    def retrieve_similar_data(self, query: str, k: int = 5) -> List[Tuple[str, str, str]]:
        """
        Retrieves the most similar questions based on the query.

        Args:
            query (str): The query question.
            k (int): Number of similar questions to return (default: 5).

        Returns:
            List[Tuple[str, str, str]]: List of tuples containing similar questions, code answers, and text answers.
        """
        if not self.vectorstore:
            raise ValueError("Vector index has not been loaded.")

        # query_embedding = self.embedding_model.embed_query(query)
        
        # Perform search in the vector store
        results = self.vectorstore.similarity_search(query, k=k)
        
        similar_items = []
        for result in results:
            question = result.page_content
            code_answer = result.metadata.get('code_answer', 'N/A')
            text_answer = result.metadata.get('text_answer', 'N/A')
            
            similar_items.append((question, code_answer, text_answer))
        
        return similar_items
