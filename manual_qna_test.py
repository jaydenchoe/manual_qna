import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# 현재 스크립트의 디렉토리를 작업 디렉토리로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

def load_vector_db(index_name, openai_api_key):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
        index_path = os.path.join(current_dir, index_name)
        if not os.path.exists(index_path):
            st.error(f"FAISS 인덱스 파일 '{index_name}'을 찾을 수 없습니다.")
            return None
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"벡터 데이터베이스 로드 중 오류 발생: {str(e)}")
        return None

def create_qa_chain(vector_db, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=1, openai_api_key=openai_api_key, max_tokens=4096)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    # top_k를 5로 설정
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

def display_source_info(source):
    st.markdown("#### 출처 상세 정보")
    st.text_area("내용", source.page_content, height=150)
    st.json(source.metadata)

def main():
    st.title("FAISS 기반 QnA 시스템")

    st.write(f"현재 작업 디렉토리: {os.getcwd()}")

    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
    if not openai_api_key:
        st.warning("OpenAI API 키를 입력해주세요.")
        return

    index_name = st.text_input("사용할 FAISS 인덱스 파일 이름을 입력하세요 (확장자 제외):", "faiss_index")
    
    vector_db = load_vector_db(index_name, openai_api_key)
    if vector_db is None:
        return

    qa_chain = create_qa_chain(vector_db, openai_api_key)

    st.success("FAISS 인덱스 로드 완료!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("질문을 입력하세요:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = qa_chain({"question": prompt})
                response = result['answer']
                st.markdown(response)

                # 출처 표시 (expander 사용)
                if 'source_documents' in result:
                    st.markdown("### 출처")
                    for i, doc in enumerate(result['source_documents']):
                        with st.expander(f"출처 {i+1}"):
                            display_source_info(doc)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
