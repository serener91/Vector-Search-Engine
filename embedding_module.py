from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List
import torch
from dotenv import load_dotenv


def embed_text(model_name: str = "BAAI/bge-m3", text: str = None):

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings.embed_query(text)


def embed_document(model_name: str = "BAAI/bge-m3", documents: List[str] = None):

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    return embeddings.embed_documents(documents)


def load_embedding_model(model_name: str = "BAAI/bge-m3"):

    if torch.cuda.is_available():
        model_kwargs = {'device': 'cuda'}

    else:
        model_kwargs = None

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, show_progress=True)

    return embeddings


def load_openai_embedding():

    load_dotenv()
    embeddings = OpenAIEmbeddings()

    return embeddings


if __name__ == '__main__':
    test_model = ["BAAI/bge-m3",
                  "sentence-transformers/msmarco-distilbert-base-v4"]
