# 01. Basic Chatbot

Streamlit과 LangChain LCEL을 사용한 기본 챗봇 실습입니다.

## 실행

루트 디렉터리에서 실행합니다.

```bash
streamlit run examples/01-basic-chatbot/app.py
```

## 주요 내용

- Streamlit 채팅 UI
- `st.session_state`를 사용한 대화 내역 저장
- `ChatPromptTemplate`, `ChatOpenAI`, `StrOutputParser` 기반의 간단한 체인
