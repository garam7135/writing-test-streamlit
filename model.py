from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import streamlit
def tokenizer(text):
    '''
    text: 책 내용 전문 텍스트
    '''

    # 스플리터 지정
    # 가장 긴 눈의 여왕 텍스트 수 : 34,000자. 
    # token 수: 약 34000자 
    # OpenAI 모델 gpt-3.5-turbo-0125 토큰 수 : 16,385 tokens -> spliter 필요
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",  # 분할기준
        chunk_size=5000,   # 사이즈
        chunk_overlap=500, # 중첩 사이즈
    )

    # 분할 실행
    split_docs = text_splitter.split_text(text)
    split_docs = [Document(page_content=text) for text in split_docs]

    # 총 분할된 도큐먼트 수
    print("문할된 문서의 수 :", str(len(split_docs)))
    print(split_docs)
    
    return split_docs

def langchain(text="책원문", interesting="재미있던 부분", impression="느낀점"):

    '''
    - 요약/ 독후감 langchain model
    '''

    ''' 입력 범위 제한'''
    text = text[:40000]
    interesting = interesting[:50]
    impression = impression[:50]

    split_docs = tokenizer(text)

    ################## 1. Map 단계: split한 문서를 각각 요약하는 단계 ##################
    # Map 프롬프트
    map_template = """다음 입력은 책 줄거리의 일부분입니다.
    {pages}
    요약에 들어가야 할 내용:
    """
    map_template+=f"""{interesting}
    책 줄거리 일부분에 대해 주요 내용을 요약해 주세요.
    - 요약에 들어가야 할 내용과 줄거리가 일치하는 부분이 있다면, 해당 내용을 반드시 포함시켜 주세요.
    - 책 줄거리에 주어진 내용만을 사용해 요약을 작성하세요. 새로운 내용을 추가하지 마세요.
    - 문맥을 자연스럽게 유지해 주세요.
    - 책 제목을 첫 번째 줄에 포함시켜주세요.
    """

    # Map 프롬프트 정의
    map_prompt = PromptTemplate.from_template(map_template)
    print(map_prompt)

    # Map LLMChain 정의
    llm = ChatOpenAI(openai_api_key =streamlit.secrets["OPENAI_API_KEY"],
                    temperature=0,
                    model_name='gpt-3.5-turbo-0125')
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    ################## 2. Reduce 단계: 각각의 문서를 하나로 요약하는 단계 ##################
    # Reduce 프롬프트
    reduce_template = """다음은 책 줄거리 요약의 리스트입니다:
    {doc_summaries}
    요약에 들어가야 할 내용:
    """
    reduce_template += f"""
    {interesting}
    위 내용을 바탕으로 자연스러운 통합된 요약을 만들어 주세요.
    - 요약에 들어가야 할 내용을 포함시켜 주세요.
    - 모든 내용을 자연스럽게 엮어 주세요.
    - 요약에 들어가야 할 내용이 너무 많거나, 자연스럽지 못한 부분이 있는지 한번 더 검토해 주세요.
    - 통합된 요약은 반드시 책 줄거리 요약의 리스트를 참조해야 합니다. 없는 내용을 만들어 내지 마세요.
    - 책 제목을 첫 번째 줄에 포함시켜주세요.
    """

    # Reduce 프롬프트 정의
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    print(reduce_prompt)

    # Reduce에서 수행할 LLMChain 정의
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    ################## 3. Map-Reduce 단계 ##################
    # 문서 리스트를 단일 문자열로 결합하고, 이를 LLMChain에 전달
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,                
        document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000 # 문서 최대 토근 수
    )

    # Map-reduce chain 정의
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain, # Map 체인
        reduce_documents_chain=reduce_documents_chain, # Reduce 체인
        document_variable_name="pages", # map-chain input 변수명
        return_intermediate_steps=False, # 중간 단계 반영 변수
    )

    # 최종 실행
    sum_result = map_reduce_chain.invoke(split_docs)

    # 요약결과 출력
    print(sum_result['output_text'])

    # 독후감 프롬프트
    report_template = """ 아래 내용을 바탕으로 독후감을 작성해 주세요.
    [책 줄거리], [재미있던 내용], [느낀점]을 반드시 아래 형식에 맞게 반영해 주세요.

    독후감 형식:

    처음:
    - 독후감 제목, 책 제목, 책을 읽은 동기
    - 독후감 제목을 정하고, 책 제목을 부제목으로 붙인 뒤, 왜 이책을 읽게 되었는지를 작성해 주세요.
    중간:
    - 책 줄거리, 기억에 남거나 감동 받은 장면, 재미있게 느낀 내용
    - 책 줄거리를 간단하게 요약하고, 기억에 남는 장면, 재미있던 내용 등을 담아 작성해 주세요.
    끝:
    - 전체적인 느낌점 및 감상, 반성한 점, 새로운 다짐이나 결심
    - 책을 읽고 난 후 들었던 생각과, 느낀 점을 정리하는 부분으로 자신만의 느낌이나 교훈 등을 담아 작성해 주세요.

    독후감은 형식을 유지하지만 처음, 중간, 끝이라는 키워드를 출력하지 마세요.

    [책 줄거리]
    {text}

    """
    report_template += f"""
    [재미있던 내용]:
    {interesting}

    [느낀점]:
    {impression}

    [생성된 독후감]:
    """
    
    # 독후감 프롬프트 정의
    report_prompt = PromptTemplate(template=report_template, input_variables=['text'])
    print(report_prompt)

    # 독후감 chain
    llm_chain = LLMChain(prompt=report_prompt, llm=llm)

    # 실행 및 결과
    bookreport_result = llm_chain.invoke(sum_result['output_text'])
    print(bookreport_result['text'])

    return sum_result['output_text'], bookreport_result['text']

if __name__ == '__main__':
    # test code
    sum_result, bookreport_result = langchain(text="인어공주는", interesting="인어 공주가 승천하는 부분", impression="착하게 살아야 한다.")
