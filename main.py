'''

독후감 및 책 요약문 생성 Test main

'''
import streamlit as st

from model import langchain_model

# 1. markdown 스타일 지정
st.markdown(
    """
    <style>
    .container {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .column {
        padding: 10px;
    }
    .small-font {
        font-size: 20px;
        text-align: center;
    }
    .ssmall-font {
        font-size: 13px;
        text-align: left;
        padding: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# main 실행부
def main():
    # streamlit으로 제목 생성
    st.title("독후감 & 요약 PoC")
    with st.container(border=True):
        st.text("주의 사항")
        st.markdown('<p class="ssmall-font">1. 입력되지 않은 값에 대해서는 랜덤으로 생성됩니다. 필요한 부분은 반드시 입력해 주세요.</p>', unsafe_allow_html=True)
        st.markdown('<p class="ssmall-font">2. 책에 존재하지 않는 내용을 입력할 시 새로운 창작 결과가 생성될 수 있습니다.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        content = st.text_area("책의 원문을 입력해 주세요. (최대 40,000자 입력 가능)", max_chars=40000)
        interesting = st.text_input("책에서 가장 재밌었던 부분을 입력해 주세요. (최대 50자 입력 가능)", max_chars=50)
        impression = st.text_input("책을 통해 느낀점을 입력해 주세요. (최대 50자 입력 가능)", max_chars=50)

    # 결과 보기 버튼 누름시, 컨테이너 안에 두 가지 output 결과를 좌우로 배치
    if st.button("결과 보기"):
        with st.container():            
            # 좌우로 나눌 컬럼 설정
            col1, col2 = st.columns(2)
            
            # 첫 번째 컬럼: 독후감 결과
            with st.spinner("독후감 및 요약문이 작성 중입니다... 책 원문 길이에 따라 최대 1~2분 정도 소요됩니다."):
                sum_result, bookreport_result = langchain_model(text=content, interesting=interesting, impression=impression)
                with col1:
                    st.markdown('<div class="column">', unsafe_allow_html=True)
                    st.markdown('<p class="small-font">독후감 자동생성</p>', unsafe_allow_html=True)
                    st.write(bookreport_result)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 두 번째 컬럼: 요약 결과
                with col2:
                    st.markdown('<div class="column">', unsafe_allow_html=True)
                    st.markdown('<p class="small-font">책 요약 결과</p>', unsafe_allow_html=True)
                    st.write(sum_result)
                    st.markdown('</div>', unsafe_allow_html=True)
            
if __name__ == '__main__':
    main()