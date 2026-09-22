# PART 04. 에이전트 — 학습 정리 개요

> 이 교재는 약 1년 전에 쓰였고, 그 사이 LangChain/LangGraph API가 여러 번 바뀌었습니다.
> 아래는 "교재에서 배운 개념"과 "2026년 9월 기준 실무에서 쓰는 방식"을 같이 정리한 것입니다.
> 개념 자체는 그대로 유효하지만, **실제로 코드를 짤 때 쓰는 함수/클래스는 달라졌습니다.**

## 문서 구성

| 파일 | 내용 |
|---|---|
| `01-tools.md` | 도구(Tool) 기초 — 빌트인 도구, 커스텀 도구 |
| `02-tool-calling-agent.md` | Tool Binding → Tool Calling Agent → AgentExecutor → 스트리밍/iter/HITL |
| `03-memory-and-multi-agent.md` | 대화 메모리, 멀티 에이전트(에이전트를 도구로 감싸기) |
| `04-agentic-rag.md` | Agentic RAG 개념과 도구 우선순위 판단 |
| `05-practical-agents.md` | CSV/Excel 분석 에이전트, 파일 관리 자동화 에이전트 |

## 가장 중요한 변화 한눈에 보기

| 교재 방식 (구버전) | 상태 (2026.09 기준) | 현재 권장 방식 |
|---|---|---|
| `AgentExecutor` | **레거시 / 유지보수 모드** (보안 패치만, 신기능 없음) | `langchain.agents.create_agent` (LangGraph 기반) |
| `create_react_agent` (langgraph.prebuilt) | **Deprecated** (LangGraph v1에서 폐기) | `langchain.agents.create_agent` |
| `RunnableWithMessageHistory` | **Deprecated 예정**, 신규 프로젝트 비권장 | LangGraph Checkpointer (단기 메모리) + BaseStore (장기 메모리) |
| `ConversationBufferMemory` 등 메모리 클래스 | v0.3.1부터 deprecated, v1.0에서 제거 예정 | LangGraph persistence |
| `create_pandas_dataframe_agent` | `langchain_experimental`에 위치, 여전히 실험적 | 개념 학습용으로는 OK, 프로덕션은 직접 구현 또는 최신 대안 확인 필요 |

### 왜 이렇게 바뀌었나?

- LangChain 자체는 "체인/도구/모델 통합" 라이브러리 성격이 강했고, **에이전트의 반복 실행·상태 관리·중단/재개(human-in-the-loop)** 같은 복잡한 제어 흐름에는 구조적 한계가 있었습니다.
- **LangGraph**가 그 역할을 전담하게 되면서, LangChain의 에이전트 관련 기능들은 점점 "LangGraph를 감싼 얇은 래퍼"로 대체되고 있습니다.
- 2026년 현재는 `langchain.agents.create_agent` 하나가 사실상 표준 진입점입니다. 내부적으로 LangGraph의 `StateGraph` 위에서 동작하며, 미들웨어(PII 보호, 대화 요약, 사람 승인 등)로 확장 가능합니다.

```python
# 2026년 기준 표준 진입점 예시
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아 날씨를 반환합니다."""
    return f"{city}는 맑음"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
)

result = agent.invoke({"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]})
```

### 학습 전략 제안

- **개념(Tool Binding, ReAct 순환, Action/Observation, Agentic RAG 등)은 그대로 유효**하니 교재로 이해하는 것 자체는 문제없습니다.
- 다만 **실습 코드를 그대로 프로덕션에 쓰지 말고**, 개념 이해 후에는 `create_agent` 기준으로 한 번씩 다시 짜보는 걸 추천합니다.
- 이력서/포트폴리오에 "AgentExecutor로 만든 프로젝트"라고 쓰기보다는, 가능하면 최신 API로 리팩터링한 버전을 보여주는 게 AX 직무 지원 시 더 좋은 인상을 줄 수 있습니다.
