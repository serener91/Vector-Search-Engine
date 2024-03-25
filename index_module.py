from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.milvus import Milvus
from typing import List
import os


def milvus_db(documents: List[str], embeddings, connection_args: dict, collection_name: str):
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
        drop_old=True)

    db_index = vector_store.from_documents(
        documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=connection_args,
    )

    return db_index


def faiss_db(documents: List[str], embeddings, save_index: bool = False, save_path: str = None, index_name: str = "pdf_index"):

    """
    Input: Document
    Output: Document
    """

    if len(os.listdir("/workspace/save_dir/db_indices")) > 0:
        db_index = FAISS.load_local("/workspace/save_dir/db_indices", embeddings, index_name)
        print("Index loaded!")
        return db_index

    else:
        db_index = FAISS.from_documents(documents, embeddings)

        if save_index:
            if save_path is None:
                save_path = os.getcwd()
                db_index.save_local(save_path, index_name)

            else:
                db_index.save_local(save_path, index_name)

            print("Index Saved!")

        return db_index



