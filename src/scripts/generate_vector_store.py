import argparse
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

def create_vectorstore(df: pd.DataFrame,
                       embedding_model_name: str = "text-embedding-ada-002",
                       index_filename: str = "vector_store/questions_index.pkl"):

    if 'Question' not in df.columns:
        raise ValueError("DataFrame must contain 'Question' column.")

    questions = df['Question'].tolist()
    documents = [Document(page_content=q, metadata={"code_answer": df.loc[i, "Code_Answer"], "text_answer": df.loc[i, "Text_Answer"]}) for i, q in enumerate(questions)]

    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=index_filename)
    print(f"Vector index with metadata saved successfully to {index_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector store from a dataset.")
    parser.add_argument(
        "--path",
        type=str,
        default="dataset/TechHazardQA_translated.json",
        help="Path to the dataset JSON file.",
    )

    args = parser.parse_args()
    df = pd.read_json(args.path, lines=True)[0:30]  # Remove slicing here to process the entire dataset
    create_vectorstore(df)
