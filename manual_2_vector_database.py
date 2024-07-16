import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import tempfile
import shutil

# 현재 스크립트의 디렉토리를 작업 디렉토리로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

st.write(f"현재 작업 디렉토리: {os.getcwd()}")

def load_and_split_pdf(pdf_file):
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # 임시 파일 삭제
    os.unlink(tmp_file_path)

    # 텍스트 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_db(chunks, openai_api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def main():
    st.title("PDF를 FAISS 데이터베이스로 변환")

    # OpenAI API 키 입력
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    if not openai_api_key:
        st.warning("OpenAI API 키를 입력해주세요.")
        return

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("PDF 처리 중..."):
            chunks = load_and_split_pdf(uploaded_file)
            st.write(f"총 {len(chunks)}개의 청크로 분할되었습니다.")
            vector_db = create_vector_db(chunks, openai_api_key)

        st.success("PDF 처리 완료!")

        # FAISS 인덱스 저장
        save_path = st.text_input("FAISS 인덱스를 저장할 경로를 입력하세요:", "faiss_index")
        
        # 기존 FAISS 데이터베이스 확인 및 삭제 옵션
        if os.path.exists(save_path):
            st.warning(f"'{save_path}' 경로에 이미 FAISS 데이터베이스가 존재합니다.")
            if st.button("기존 데이터베이스 삭제"):
                shutil.rmtree(save_path)
                st.success(f"기존 FAISS 데이터베이스가 삭제되었습니다.")

        if st.button("저장"):
            vector_db.save_local(save_path)
            st.success(f"FAISS 인덱스가 {save_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
