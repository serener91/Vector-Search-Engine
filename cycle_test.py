from chunking_module import *
from embedding_module import load_embedding_model, load_openai_embedding
from reranking_module import rerank_texts
from index_module import *
from llm_module import call_openai, call_hf_model


def web_search(data, question):

    raw_texts = load_webpage(data)
    text_list = [text["text"] for text in raw_texts]
    document_chunks = text_chunking(documents=text_list)

    embeddings = load_embedding_model()
    # embeddings = load_openai_embedding()

    vs = faiss_db(document_chunks, embeddings)
    retriever = vs.as_retriever().get_relevant_documents(question)
    retrieved_docs = [text_chunk.page_content for text_chunk in retriever]
    # print(len(retrieved_docs))
    rerank_docs = rerank_texts(question=question, documents=retrieved_docs)
    # print(len(rerank_docs))

    # response = call_openai(rerank_docs, question)
    response = call_hf_model(texts=rerank_docs, question=question)

    return response


def pdf_search(data, question):

    raw_texts = load_pdf(data)
    text_list = [text["text"] for text in raw_texts]
    document_chunks = text_chunking(documents=text_list)

    embeddings = load_embedding_model()
    # embeddings = load_openai_embedding()

    vs = faiss_db(document_chunks, embeddings, save_index=True, save_path="/workspace/save_dir/db_indices", index_name="pdf_index")
    retriever = vs.as_retriever().get_relevant_documents(question)
    retrieved_docs = [text_chunk.page_content for text_chunk in retriever]
    # print(len(retrieved_docs))
    rerank_docs = rerank_texts(question=question, documents=retrieved_docs)
    # print(len(rerank_docs))

    response = call_openai(rerank_docs, question)
    # response = call_hf_model(texts=rerank_docs, question=question)

    return response


if __name__ == '__main__':
    # wps =  ["https://n.news.naver.com/mnews/article/028/0002680313?sid=101",
    #         "https://n.news.naver.com/mnews/hotissue/article/648/0000023897?type=series&cid=2000034"]
    # print(web_search(wps, "금 투자방법에 대해서 알려줘"))

    pdf_file = [os.path.join("/workspace/data", filename) for filename in os.listdir("/workspace/data")]
    print(pdf_search(pdf_file, "직장 내 괴롭힘 예방지침 제 2조에는 무슨 내용이 적혀있어?"))
    print(pdf_search(pdf_file, "정기적 검사에서 연차검사는 어느 규정을 따르는가?"))
    print(pdf_search(pdf_file, "단저구조에서 중심선 내용골 구조와 배치는 어떻게 해야돼? "))















