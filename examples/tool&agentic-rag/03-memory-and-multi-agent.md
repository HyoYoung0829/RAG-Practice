# 03. 대화 메모리 & 멀티 에이전트

## 대화 메모리 (교재 방식)

교재에서는 이전 대화 내용을 기억하는 에이전트를 만들기 위해 두 가지를 사용했다:

- **`ChatMessageHistory`**: 대화 메시지(사람/AI)를 순서대로 저장하는 객체
- **`RunnableWithMessageHistory`**: 체인/에이전트를 감싸서, 매 호출마다 이전 대화 기록을 자동으로 불러오고 저장해주는 래퍼

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_history.invoke(
    {"input": "내 이름은 주인이야"},
    config={"configurable": {"session_id": "user1"}},
)
```

- `session_id`로 대화방을 구분한다 — 여러 사용자/대화 스레드를 동시에 관리 가능.

## 2026년 기준 업데이트: 메모리

**`RunnableWithMessageHistory`는 신규 프로젝트에 비권장**이다 (deprecated 수순). `ConversationBufferMemory` 등 구버전 메모리 클래스는 이미 deprecated 되었고 v1.0에서 제거 예정.

현재 권장 방식은 **LangGraph의 Persistence(영속성) 시스템**이다:

| 개념 | 역할 |
|---|---|
| **Checkpointer** | 하나의 대화 스레드(thread) 안에서의 단기 메모리 — `session_id` 개념과 비슷 |
| **BaseStore** | 여러 대화 스레드를 넘나드는 사용자 단위 장기 메모리 (예: "이 사용자는 항상 존댓말 선호") |

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

checkpointer = MemorySaver()
agent = create_agent(model=llm, tools=tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user1"}}
agent.invoke({"messages": [{"role": "user", "content": "내 이름은 주인이야"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐라고 했지?"}]}, config)
```

- `session_id` → `thread_id`로 개념 명칭만 바뀌었다고 이해하면 편함
- 장점: 에러 복구, 특정 시점으로 되돌리기("타임 트래블"), 멀티 유저 지원이 구조적으로 쉬움
- 실무에서 대화가 길어져 컨텍스트가 넘칠 때를 위한 요약(summarization) 미들웨어도 표준으로 제공됨

## 여러 에이전트 조합 (에이전트를 도구로 감싸기)

교재에서 "이해 안 갔다"고 한 부분 — 개념은 이거다:

**하나의 에이전트(서브 에이전트)를 통째로 다른 에이전트의 도구(Tool)처럼 취급**하는 패턴.

```python
# 서브 에이전트: 데이터 분석 전용
analysis_agent = create_tool_calling_agent(llm, [pandas_tool], prompt)
analysis_executor = AgentExecutor(agent=analysis_agent, tools=[pandas_tool])

# 이걸 함수로 감싸서 "도구"처럼 만듦
@tool
def run_data_analysis(question: str) -> str:
    """데이터 분석이 필요할 때 사용하는 전문 에이전트."""
    return analysis_executor.invoke({"input": question})["output"]

# 메인 에이전트는 이 분석 에이전트를 하나의 도구로 씀
main_agent = create_tool_calling_agent(llm, [run_data_analysis, other_tool], prompt)
```

**왜 이렇게 하나?**
- 하나의 에이전트가 모든 역할(검색, 분석, 파일 관리 등)을 다 하게 만들면 프롬프트가 비대해지고 판단 정확도가 떨어짐
- 역할별로 **전문화된 서브 에이전트**를 만들고, 메인 에이전트는 "어떤 전문가에게 넘길지"만 판단하게 하면 훨씬 안정적
- 이게 바로 **멀티 에이전트 오케스트레이션**의 기본 아이디어 — "Supervisor(감독) 에이전트가 여러 Worker(작업자) 에이전트에게 일을 나눠주는" 구조와 같은 개념

## 2026년 기준 업데이트: 멀티 에이전트

이 패턴은 오히려 **지금 더 주류가 된 개념**이다. LangGraph는 이를 위한 전용 기능을 제공한다:

- **Handoff(핸드오프)**: 에이전트 간에 작업을 명시적으로 넘기는 패턴
- **Supervisor 패턴**: 하나의 감독 에이전트가 여러 전문 서브 에이전트를 도구처럼 호출
- Anthropic/LangChain 생태계에서도 "여러 전문 에이전트 + 감독 에이전트" 구조가 복잡한 업무 자동화의 표준 설계로 자리잡음

교재에서 이해가 안 갔던 부분이 사실 **지금 실무에서 제일 중요하게 다뤄지는 패턴 중 하나**이니, 이 개념은 나중에 LangGraph의 multi-agent 문서로 한 번 더 짚고 넘어가는 걸 추천.
