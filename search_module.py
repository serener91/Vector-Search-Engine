from sentence_transformers import SentenceTransformer, util
from typing import List



def search_texts(model_name: str = "BAAI/bge-m3", top_k: int = 5 , question: str = None, documents: List[str] = None):

    """
    Input: List of Texts
    Output: List of Texts
    """

    bi_encoder = SentenceTransformer(model_name)
    # print(bi_encoder.max_seq_length)
    # bi_encoder.max_seq_length = 512

    question_embedding = bi_encoder.encode(question, convert_to_tensor=True).to("cuda")
    document_embeddings = bi_encoder.encode(documents, convert_to_tensor=True, show_progress_bar=True).to("cuda")

    results = util.semantic_search(question_embedding, document_embeddings, top_k=top_k)[0]
    doc_ids = [i["corpus_id"] for i in results]
    searched_documents = [documents[doc_id] for doc_id in doc_ids]
    # output = "\n".join(searched_documents)

    return searched_documents



if __name__ == '__main__':
    query = "의사들 무슨 문제가 있는거야?"
    passages = [
        "정부가 전국 40개 대학을 대상으로 시행한 의대 정원 수요조사 마감 다음 날인 5일 의과대학 교수들이 사직서를 제출하거나 삭발식을 여는 등 반발하고 있다.",
        "전공의 집단 사직 사태가 보름째 계속되는 가운데, 현장에 남은 의료진이 업무 과중으로 인한 피로를 호소하거나 환자 불편이 이어지는 등 의료공백이 점차 커지는 모습이다",
        "이틀째 수련병원 현장점검을 이어가고 있는 정부는 전공의 7천여 명에 대한 미 복귀 증거를 확보했다며, 이들에 대한 행정·사법 처리에 속도를 내고 있다.",
        "더불어민주당 비명(비이재명)계 홍영표(4선·인천 부평을) 의원은 5일 백척간두에 선 심정으로 내일은 입장을 밝히려 한다며 탈당 의사를 거듭 내비쳤다.",
        " 배우 이선균씨를 협박해 금품을 뜯은 전직 영화배우는 불법 유심칩을 사용하는 등 자신의 신분을 철저하게 숨긴 채 범행한 것으로 드러났다.",
        "연구소는 감사보고서를 통해 파악한 삼성전자의 2023년도 직원 인건비(급여·퇴직급여) 총액은 14조6천778억원이었고, 이를 토대로 조만간 사업보고서에 공시될 임직원 급여 총액을 역산출한 결과 14조3천800억∼14조7천500억원 수준으로 예상된다고 설명했다.",
        "해병대 채상병 순직 사건 수사 외압 의혹으로 고발된 이종섭 전 국방부 장관이 주호주대사로 임명되면서 사건을 맡은 고위공직자범죄수사처(공수처)가 수사 방향을 고심하고 있다."
    ]

    print(search_texts(question=query, documents=passages))

    print(search_texts(model_name="BAAI/bge-m3", question=query, documents=passages))