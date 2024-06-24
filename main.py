'''

독후감 및 책 요약문 생성 PoC main

'''
import streamlit as st

from model import langchain

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
    </style>
    """,
    unsafe_allow_html=True
)
# def get_book_file_list(folder_path=""):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('선택 가능한 도서 리스트', filenames)
#     return selected_filename

def main():
    # streamlit으로 제목과 input box 생성
    st.title("독후감 & 요약 PoC")
    with st.container(border=True):
        # selected_filename = get_book_file_list(folder_path=BOOK_TEXT_PATH)
        # st.write('선택한 책 이름: `%s`' % selected_filename)
        # if st.checkbox('책 원문 내용 확인하기') :
        #     book_text = codecs.open(os.path.join('./book_text', selected_filename)).read()
        #     st.write(book_text)
        # else :
        #     st.text('원문 숨기기 상태.')
        content = st.text_area("책의 원문 내용을 입력해 주세요. (최대 35000자 입력 가능)")
        interesting = st.text_input("책에서 가장 재밌었던 부분을 입력해 주세요. (최대 50자 입력 가능)")
        impression = st.text_input("책을 통해 느낀점을 입력해 주세요. (최대 50자 입력 가능)")

    # 컨테이너 안에 두 가지 output 결과를 좌우로 배치
    if st.button("결과 보기"):
        with st.container():            
            # 좌우로 나눌 컬럼 설정
            col1, col2 = st.columns(2)
            
            # 첫 번째 컬럼: 독후감 결과
            with st.spinner("독후감 및 요약문이 작성 중입니다... 원문 내용에 따라 최대 1~2분 정도 소요됩니다."):
                sum_result, bookreport_result = langchain(text=content, interesting=interesting, impression=impression)
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
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()