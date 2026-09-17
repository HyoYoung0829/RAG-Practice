from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper


class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사 정보")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


load_dotenv()

st.title("Email 요약기 💬")


# 처음 1번만 실행
if "messages" not in st.session_state:
    # 대화 히스토리 저장 공간
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_email_parsing_chain():
    # PydanticOutputParser 생성
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """)

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # 체인 생성
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | output_parser

    return chain


def create_report_chain():
    prompt_path = Path(__file__).resolve().parent / "prompt" / "email.yaml"
    prompt = load_prompt(str(prompt_path), encoding="utf-8")

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | ChatOpenAI(model="gpt-4.1") | output_parser

    return chain


# 초기화 버튼 로직
if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 유저 입력
    st.chat_message("user").write(user_input)

    # 이메일을 파싱 chain 생성
    email_chain = create_email_parsing_chain()
    # email 에서 주요 정보를 추출
    answer = email_chain.invoke({"email_conversation": user_input})

    # 보낸 사람의 추가 정보 수집(검색)
    params = {"engine": "google", "gl": "kr", "hl": "ko", "num": "3"}  # 검색 파라미터
    search = SerpAPIWrapper(params=params)  # 검색 객체 생성
    search_query = f"{answer.person} {answer.company} {answer.email}"  # 검색 쿼리
    search_result = search.run(search_query)  # 검색 실행

    # 이메일 요약 리포트 생성
    report_chain = create_report_chain()
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }

    # 스트리밍 호출
    response = report_chain.stream(report_chain_input)
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화 히스토리 저장.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
