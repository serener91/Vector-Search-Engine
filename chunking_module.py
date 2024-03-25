from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os
from typing import List


def load_pdf(filepaths: List[str] = None):

    """Extract text from .pdf by page"""

    contents = []
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        for page in pages:
            contents.append(
            {"text": page.page_content,
             "source": os.path.basename(page.metadata["source"]),
             "page": page.metadata["page"] + 1}
            )

    return contents


def load_webpage(webpages: List[str]):

    loader = WebBaseLoader(webpages)
    pages = loader.load()
    contents = []
    for page in pages:
        contents.append(
        {"text": page.page_content,
         "source": page.metadata["source"],
         "title": page.metadata["title"]}
        )

    return contents


def text_chunking(documents: List[str] = None, chunk_size: int = 1000, overlap: float = 0.1):

    """
    Divide texts by length
    Input: Document
    Output: Document

    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        # chunk_overlap=round(chunk_size * overlap),
        length_function=len,
        # add_start_index=True,
        # is_separator_regex=False,
    )

    text_chunks = text_splitter.create_documents(documents)

    return text_chunks



def token_chunking(model_name: str = None, documents: List[str] = None, chunk_size: int = 1000, overlap: float = 0.1):

    """
    Divide texts by tokens
    Input: Document
    Output: Document
    """

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True),
        chunk_size=chunk_size,
        # chunk_overlap=round(chunk_size * overlap),
    )

    text_chunks = text_splitter.create_documents(documents)

    return text_chunks



if __name__ == '__main__':
    texts = load_pdf(["/workspace/data/직장 내 괴롭힘 예방지침 개정 전문.pdf"])
    page_contents = [text["text"] for text in texts]
    chunks = text_chunking(documents=page_contents)
    print(chunks)
    passages = [text_chunk.page_content for text_chunk in chunks]
    print(passages)
    print(len(passages))

    tok_chunks = token_chunking(model_name="davidkim205/komt-mistral-7b-v1", documents=page_contents)
    print(tok_chunks)
    tok_passages = [text_chunk.page_content for text_chunk in tok_chunks]
    print(tok_passages)
    print(len(tok_passages))

    import pandas as pd

    a = pd.concat([pd.DataFrame(passages), pd.DataFrame(tok_passages)], axis=1)
    print(a)



