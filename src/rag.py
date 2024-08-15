import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.core.utils import read_pickle

class RAG:
    def __init__(self, 
                 embedding_model_name: str = "text-embedding-3-small", 
                 index_filename: str = "vector_store/questions_index.pkl"):
        """
        Initializes the RAG class.

        Args:
            embedding_model_name (str): Name of the embeddings model (default: "text-embedding-3-small").
            index_filename (str): Filename to load the vector index (default: "questions_index.pkl").
        """
        self.embedding_model_name = embedding_model_name
        self.index_filename = index_filename
        self.vectorstore = None
        self._load_index()

    def _load_index(self):
        """
        Loads the vector index from a file.
        """
        if os.path.exists(self.index_filename):
            try:
                self.vectorstore = read_pickle(self.index_filename)
                print("Vector index loaded successfully.")
            except Exception as e:
                print(f"Error loading vector index: {e}")
                raise
        else:
            raise FileNotFoundError("Vector index file does not exist.")

    def retrieve_similar_questions(self, query: str, k: int = 5) -> List[Tuple[str, str, str]]:
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

        embedding_model = OpenAIEmbeddings(model_name=self.embedding_model_name)
        query_embedding = embedding_model.embed(query)
        results = self.vectorstore.search(query_embedding, k=k)
        indices = results[1]
        
        # Retrieve additional information stored in the vectorstore
        similar_items = [(self.vectorstore.documents[i], 
                          self.vectorstore.additional_info[i][0], 
                          self.vectorstore.additional_info[i][1]) 
                         for i in indices]
        return similar_items
