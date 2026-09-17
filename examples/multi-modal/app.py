import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from multimodal import MultiModal

from dotenv import load_dotenv
import os

load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 저장 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("이미지 인식 기반 챗봇 💬")

# 대화 내역 저장하는 리스트
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이미지, 대화 탭 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])


# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    # 모델 선택
    selected_model = st.selectbox("LLM 선택", ["gpt-4o-mini"], index=0)

    # 시스템 프롬프트 입력
    system_prompt = st.text_area(
        "시스템 프롬프트",
        "당신은 명화를 해석하는 AI 어시스턴트 입니다.\n당신의 임무는 주어진 랜덤한 명화를 바탕으로 작가의 의도를 추측하여 친절하게 답변하는 것입니다.",
        height=200,
    )


# 대화 출력 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)


# 대화 저장 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지 저장 함수
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 이미지 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 답변 생성 함수
def generate_answer(
    image_filepath, system_prompt, user_prompt, model_name="gpt-4o-mini"
):
    # 모델 생성
    llm = ChatOpenAI(
        temperature=0,
        model=model_name,
    )

    # 멀티모달 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 기반 답변 스트리밍
    answer = multimodal.stream(image_filepath)
    return answer


# 초기화 버튼 로직
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지 영역
warning_msg = main_tab2.empty()

# 이미지 업로드 처리
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)

# 질문 입력 처리
if user_input:
    # 이미지 업로드 확인
    if uploaded_file:
        # 이미지 저장
        image_filepath = process_imagefile(uploaded_file)
        # 답변 요청
        response = generate_answer(
            image_filepath, system_prompt, user_input, selected_model
        )

        # 사용자 질문 출력
        main_tab2.chat_message("user").write(user_input)

        with main_tab2.chat_message("assistant"):
            # 스트리밍 처럼 보이게 하기 위해서
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 히스토리에 대화 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 이미지 미업로드 경고
        warning_msg.error("이미지를 업로드 해주세요.")
