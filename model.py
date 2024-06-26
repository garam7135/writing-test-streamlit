from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import streamlit
def tokenizer(text):
    '''
    text: 책 내용 전문 텍스트
    '''

    # 스플리터 지정
    # 가장 긴 눈의 여왕 텍스트 수 : 34,000자. 
    # token 수: 약 35,000
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

def langchain_model(text="책 줄거리 랜덤 생성", interesting="재미있던 부분 랜덤 생성", impression="느낀점 랜덤 생성"):

    '''
    - 요약/ 독후감 langchain model
    '''

    split_docs = tokenizer(text)
    llm = ChatOpenAI(api_key=streamlit.secrets["OPENAI_API_KEY"],
                    temperature=0,
                    model_name='gpt-3.5-turbo-0125')

    ''' 1. Map 단계 '''
    # Map 프롬프트
    map_template = """다음 입력은 책 줄거리의 일부분입니다.
    {pages}
    요약에 들어가야 할 내용:
    """
    map_template+=f"""{interesting}
    책 줄거리 일부분에 대해 주요 내용을 요약해 주세요.
    - 요약에 들어가야 할 내용이 줄거리에 있다면, 해당 내용을 반드시 포함시켜 주세요.
    - 책 줄거리에 주어진 내용만을 사용해 요약을 작성하세요. 책 줄거리에 없는 내용은 생성하지 말아야 합니다.
    - 문맥이 자연스러운지 한번 더 확인해 주세요.
    - 첫 줄에 책 제목을 포함해 주세요.
    """

    # Map 프롬프트 정의
    map_prompt = PromptTemplate.from_template(map_template)
    print(map_prompt)

    # Map LLMChain 정의
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    ''' 2. Reduce 단계: 각각의 문서를 하나로 요약하는 단계 '''
    # Reduce 프롬프트
    reduce_template = """다음은 책 줄거리 요약의 리스트입니다:
    {doc_summaries}
    요약에 들어가야 할 내용:
    """
    reduce_template += f"""
    {interesting}
    위 내용을 바탕으로 자연스러운 통합된 요약을 만들어 주세요.
    - 책의 중요한 줄거리들을 반영하여 내용을 자세하게 요약해 주세요.
    - 요약에 들어가야 할 내용이 줄거리에 있다면, 해당 내용을 반드시 포함시켜 주세요.
    - 책 줄거리에 주어진 내용만을 사용해 요약을 작성하세요. 책 줄거리에 없는 내용은 생성하지 말아야 합니다.
    - 문맥이 자연스러운지 한번 더 확인해 주세요.
    - 첫 줄에 책 제목을 포함해 주세요.
    - 최소 300자 ~ 최대 500자 이내로 작성해 주세요.
    """

    # Reduce 프롬프트 정의
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    print(reduce_prompt)

    # Reduce에서 수행할 LLMChain 정의
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    ''' 3. Map-Reduce 단계'''
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
    입력된 책 줄거리, 재미있던 내용, 느낀점을 반드시 독후감 형식에 적절하게 반영해 주세요.

    독후감 형식:
    1.
    - 요소: 독후감 제목, 책 제목, 책을 읽은 동기
    - 설명: 독후감 제목을 정하고, 책 제목을 부제목으로 붙인 뒤, 이책을 왜 읽게 되었는지를 작성해 주세요.
    2.
    - 요소: 책 줄거리, 기억에 남거나 감동 받은 장면, 재미있게 느낀 내용
    - 설명: 책 줄거리를 간단하게 요약하고, 기억에 남는 장면, 재미있던 내용 등을 담아 작성해 주세요.
    3.
    - 요소: 전체적인 느낌점 및 감상, 반성한 점, 새로운 다짐이나 결심
    - 설명: 책을 읽고 난 후 들었던 생각과, 느낀 점을 정리하는 부분으로 자신만의 느낌이나 교훈 등을 담아 작성해 주세요.

    위 내용들을 순서대로 작성하여 하나의 독후감을 만들어 주세요.
    번호, 요소, 형식 등은 출력하지 마세요.

    책 줄거리:
    {text}

    """
    report_template += f"""
    재미있던 내용:
    {interesting}

    느낀점:
    {impression}

    생성된 독후감:
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
    # 생성결과 test
    sum_result, bookreport_result = langchain_model(text="책 줄거리 랜덤 생성", interesting="재미있던 부분 랜덤 생성", impression="느낀점 랜덤 생성")
