import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("PDF 기반 QA💬")

# 처음 1번만 실행
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

if "document_id" not in st.session_state:
    st.session_state["document_id"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=2
    )


# 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# PDF를 벡터DB로 만들어 retriever 반환하는 함수.
def embed_file(file_content):
    with TemporaryDirectory() as directory:
        file_path = Path(directory) / "document.pdf"
        file_path.write_bytes(file_content)
        docs = PDFPlumberLoader(str(file_path)).load()

    # 단계 2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    if not split_documents:
        raise ValueError(
            "PDF에서 텍스트를 찾지 못했습니다. 스캔본은 OCR 처리가 필요합니다."
        )

    print(f"\nPDF 분할 결과: 총 {len(split_documents)}개 청크", flush=True)
    for index, document in enumerate(split_documents, start=1):
        page = document.metadata.get("page", 0) + 1
        print(
            f"\n{'=' * 60}\n"
            f"청크 {index} | 페이지 {page} | {len(document.page_content)}자\n"
            f"{'-' * 60}\n{document.page_content}",
            flush=True,
        )

    # 단계 3: 임베딩
    embeddings = OpenAIEmbeddings(request_timeout=60, max_retries=1)

    # 단계 4: DB 생성 및 저장
    # 벡터스토어 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 리트리버 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def format_documents(documents):
    return "\n\n".join(
        f"[페이지 {document.metadata.get('page', 0) + 1}]\n{document.page_content}"
        for document in documents
    )


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성
    prompt_path = Path(__file__).resolve().parent / "prompt" / "pdf-rag.yaml"
    prompt = load_prompt(str(prompt_path), encoding="utf-8")

    # 단계 7: 모델
    llm = ChatOpenAI(model=model_name, temperature=0, timeout=60, max_retries=1)

    # 단계 8: 체인 생성
    chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때 (수동 캐싱)
if uploaded_file:
    file_content = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_content).hexdigest()
    if document_id != st.session_state["document_id"]:
        st.session_state["chain"] = None
        st.session_state["messages"] = []
        st.session_state["document_id"] = None
        st.session_state.pop("retriever", None)
        try:
            with st.spinner("PDF를 처리 중입니다..."):
                st.session_state["retriever"] = embed_file(file_content)
            st.session_state["document_id"] = document_id
        except Exception as error:
            st.error(f"PDF 처리에 실패했습니다: {error}")
    if st.session_state["document_id"] == document_id:
        try:
            st.session_state["chain"] = create_chain(
                st.session_state["retriever"], model_name=selected_model
            )
            st.sidebar.success("PDF 처리 완료")
        except Exception as error:
            st.session_state["chain"] = None
            st.error(f"질문 준비에 실패했습니다: {type(error).__name__}: {error}")
else:
    if st.session_state["document_id"] is not None:
        st.session_state["messages"] = []
    st.session_state["document_id"] = None
    st.session_state["chain"] = None
    st.session_state.pop("retriever", None)

# 초기화 버튼 로직
if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 유저 입력
user_input = st.chat_input(
    "PDF에 대해 질문해 주세요!", disabled=st.session_state["chain"] is None
)

# 경고 메시지
warning_msg = st.empty()

# 유저 입력시
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            try:
                with st.spinner("답변을 생성 중입니다..."):
                    for token in chain.stream(user_input):
                        ai_answer += token
                        container.markdown(ai_answer)
                if not ai_answer.strip():
                    raise ValueError("모델이 빈 답변을 반환했습니다. 다시 질문해 주세요.")
            except Exception as error:
                st.error(f"답변 생성에 실패했습니다: {type(error).__name__}: {error}")
                st.stop()

        # 대화 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력.
        warning_msg.error("파일을 업로드 해주세요.")
