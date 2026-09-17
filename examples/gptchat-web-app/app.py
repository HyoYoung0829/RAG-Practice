import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.title("나만의 불량한 챗GPT 만들기")

# 대화 내역 저장하는 리스트
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")


# 대화 저장 함수
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 대화 출력 함수
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 체인 생성 함수
def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 아주 불량한 AI 어시스턴트입니다. 말투도 아주 시니컬하고 매 순간 화가 나있습니다.",
            ),
            ("user", "Question:\n{question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


# 초기화 버튼 로직
if clear_btn:
    st.session_state["messages"] = []

print_message()


user_input = st.chat_input("궁금한 내용을 물어보세요!")


if user_input:
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain()
    # 체인에 유저 질문 넣고 응답 담기
    response = chain.stream({"question": user_input})
    # 스트리밍 처럼 보이게 하기 위해서
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 히스토리에 대화 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
