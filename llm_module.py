from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from typing import List
from model.inference import model_response


def call_openai(texts: List[str], question: str):

    load_dotenv()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Answer must be in the same language as the question.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""

    rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | rag_prompt
            | llm
    )

    response = rag_chain.invoke({"context": "".join(texts), "question": question})

    return response.content


def call_hf_model(model_name: str = "davidkim205/komt-mistral-7b-v1", texts: List[str] = None, question: str = None):

    prompt = f"""당신은 주어진 정보를 기반으로 질문에 대답하는 업무 지원서비스를 제공합니다. 주어진 정보를 이용하여 자세한 답변을 반드시 '입니다.' 형태로 제공하세요.   
    [INST] [정보] 광학에서 분산 프리즘은 빛을 분산시키는 데 사용되는 광학 프리즘으로, 빛을 스펙트럼 구성 요소(무지개 색)로 분리하는 데 사용됩니다. [/정보] [질문] 분산형 프리즘이란 무엇인가요? [/질문]
    [/INST]분산 프리즘은 빛의 다양한 파장을 다양한 각도로 분산시키는 광학 프리즘입니다.
    [INST] [정보] {"".join(texts)} [/정보] [질문] {question} [/질문]
    [/INST]"""

    response = model_response(model_name=model_name, input_prompt=prompt)

    return response

