# 06. 추가 질문 정리

## 1. `create_agent`를 앞으로 더 자주 쓰게 되나? Agentic RAG가 실무 주류인가?

**두 질문이 섞여 있어서 나눠서 답할게.**

### `.invoke()` vs `create_agent`

이건 대체 관계가 아니라 **다른 레이어**야.

- `create_agent`: 에이전트 **객체를 만드는** 함수 (기존 `AgentExecutor(agent=..., tools=...)`가 하던 역할)
- `.invoke()`: 그렇게 만든 객체를 **실행하는** 메서드 — 이건 그대로 남아있음

```python
agent = create_agent(model=llm, tools=tools)   # 객체 생성 (AgentExecutor 대체)
agent.invoke({"messages": [...]})               # 실행 (이건 예전이랑 똑같이 씀)
```

즉 "`invoke`보다 `create_agent`를 더 쓴다"가 아니라, **에이전트를 만들 때는 `create_agent`, 실행할 때는 그 위에서 `.invoke()`/`.stream()`을 여전히 쓰는 구조**야.

### Agentic RAG가 나이브 RAG보다 실무에서 더 많이 쓰이나?

**"항상 더 많이 쓴다"는 아니고, 용도에 따라 갈려.**

| 상황 | 선택 |
|---|---|
| 단일 문서/좁은 도메인 FAQ 챗봇, 응답 속도·비용이 중요 | **나이브 RAG**가 더 나음 — 검색 1번, LLM 호출 1~2번으로 끝나서 빠르고 저렴하고 예측 가능 |
| 여러 데이터 소스(내부 문서 + 웹 + DB)를 섞어야 함 | **Agentic RAG** — 소스 선택 판단이 필요하니까 |
| 질문이 복잡해서 한 번의 검색으로 답 안 나옴 (multi-hop) | **Agentic RAG** |
| 초기 MVP, 일단 빠르게 동작하는 걸 보여줘야 함 | **나이브 RAG**로 시작하고 필요하면 Agentic으로 확장 |

실무에서는 "나이브 RAG로 시작 → 한계 보이면 Agentic RAG로 확장"이 일반적인 순서야. Agentic RAG는 LLM 호출이 늘어나서(판단 + 생성) 비용·레이턴시가 커지니까, 필요 없는데 무조건 Agentic으로 만드는 건 오히려 안티패턴으로 취급돼. 요즘 트렌드가 "에이전트"인 건 맞지만, "모든 RAG를 에이전트화해야 한다"는 아님.

---

## 2. Tool Calling Agent 구조, 더 디테일하게

메시지 리스트가 실제로 어떻게 쌓이는지 순서대로 보면 이해가 빠를 거야.

```
[1] SystemMessage: "너는 도구를 쓸 수 있는 어시스턴트야"
[2] HumanMessage: "서울 날씨 알려줘"

    ↓ LLM 호출 (1번째)

[3] AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"city": "서울"}, "id": "call_abc"}
    ])
    # LLM은 여기서 "이 도구 써야겠다"는 요청만 만듦. 실행 안 함.

    ↓ 실행기가 tool_calls를 보고 실제 함수 실행

[4] ToolMessage(tool_call_id="call_abc", content="서울: 맑음, 22도")
    # 실행 결과를 "도구 메시지"로 다시 메시지 리스트에 추가

    ↓ 이 전체 메시지 리스트(1~4)를 다시 LLM에게 통째로 넘김 (2번째 호출)

[5] AIMessage(content="서울은 맑고 22도예요.")
    # 이번엔 tool_calls가 없음 → "최종 답변"으로 판단하고 종료
```

**핵심 원리 3가지:**

1. **에이전트는 상태(메시지 리스트)를 계속 누적시키면서 LLM을 반복 호출하는 것뿐**이다. 매번 "지금까지의 전체 대화 + 도구 실행 결과"를 통째로 다시 LLM에 넣는다. (LLM 자체는 이전 호출을 기억 못 하니까)
2. **종료 조건은 단순함**: LLM의 응답에 `tool_calls`가 없으면 → 최종 답변으로 간주하고 루프 종료. 있으면 → 그 도구들을 실행하고 다시 루프.
3. 여러 도구를 한 번에 요청할 수도 있음 (`tool_calls` 리스트에 여러 개) — 이 경우 병렬로 다 실행하고, 각각 `ToolMessage`로 결과를 붙여서 다시 넘김.

이걸 "자동으로 반복해주는 것"이 `AgentExecutor`(구) / `create_agent`(신)의 역할이야. 두 구현체 모두 원리는 위와 완전히 동일하고, 차이는 **내부 구현이 파이썬 for문 기반이냐(AgentExecutor), LangGraph 그래프 기반이냐(create_agent)** 뿐이야.

---

## 3~4. Human-in-the-Loop — 직접 구현해야 하나? (구버전 / LangGraph 버전 둘 다)

### 패턴 자체는 맞음

"비용/리스크 큰 작업 전에 실행을 멈추고 사람 승인을 받는다"는 패턴이 맞고, 직접 코드로 그 분기를 만들어야 하는 것도 맞음. 다만 **"직접 구현"의 수준이 버전마다 다름**.

### AgentExecutor.iter() 방식 (구버전) — 진짜로 다 직접 짜야 함

```python
for step in agent_executor.iter({"input": "..."}):
    if "intermediate_step" in step:
        action, _ = step["intermediate_step"]
        if action.tool == "send_email":
            if input("승인? (y/n): ") != "y":
                break
```
if문, 입력 대기, 중단 로직을 전부 개발자가 손으로 짜야 함. 상태 저장(나중에 다시 이어서 실행)도 직접 구현해야 해서 번거로움.

### `create_agent` + `HumanInTheLoopMiddleware` 방식 (현재) — 설정으로 대체됨

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[send_email, search_web],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,   # 승인/수정/거절 다 허용 — 실행 전 무조건 멈춤
                "search_web": False,  # 자동 승인, 멈추지 않음
            },
        ),
    ],
    checkpointer=InMemorySaver(),  # 중단 상태를 저장하려면 필수
)
```

**여기서 개발자가 직접 하는 일:**
- `interrupt_on`에 "어떤 도구가 승인이 필요한지" **설정만 함** (if문을 직접 안 짬)
- 실행이 멈춘 뒤, 사람의 실제 승인/거절 입력을 받아서 재개시키는 로직(웹 UI 버튼, API 엔드포인트 등)은 **여전히 앱 쪽에서 직접 만들어야 함** — 이건 프레임워크가 대신해줄 수 없는 부분(승인 UI는 서비스마다 다르니까)

**즉 요약하면:**
- "멈춰야 할 시점 판단 + 상태 저장/재개" → 프레임워크(`interrupt()` 함수 + Checkpointer)가 대신 처리
- "어떤 도구가 위험한지 설정 + 실제 승인 UI/흐름" → 여전히 개발자 몫

더 세밀한 제어가 필요하면(복잡한 승인 플로우, 단계별 분기) LangGraph의 `interrupt()` 함수를 커스텀 노드 안에서 직접 호출해서 그래프를 원하는 지점에서 멈추게 만들 수도 있어 — `HumanInTheLoopMiddleware`는 그걸 표준 케이스(도구 실행 전 승인)에 맞게 미리 포장해둔 것뿐이야.

---

## 5. 대화 메모리 / 멀티 에이전트 — 그냥 "LangGraph로 넘어갔구나" 하고 넘어가도 되나?

**개념 수준에서는 맞아, 지금 단계에선 그 정도로 충분해.** 다만 나중에 실제로 구현할 때 헷갈리지 않게 이 두 개념 명칭 대응만 기억해두면 좋음:

| 구버전 개념 | LangGraph 개념 |
|---|---|
| `session_id` | `thread_id` |
| `ChatMessageHistory` (대화방 하나) | `Checkpointer`가 관리하는 스레드 상태 |
| 여러 대화방을 넘나드는 "사용자 프로필" 기억 (직접 구현해야 했음) | `BaseStore` (장기 메모리, 프레임워크가 구조 제공) |

멀티 에이전트도 "에이전트를 도구로 감싼다"는 아이디어 자체는 그대로고, LangGraph는 그걸 **Supervisor 패턴**이라는 이름으로 정식 지원한다는 정도만 알아두면 돼. 지금 단계에서 더 깊게 안 파도 됨 — 나중에 실제로 멀티 에이전트 프로젝트 할 때 그때 가서 제대로 봐도 늦지 않아.

---

## 6. Agentic RAG의 "필요하면 반복"은 실제로 어떻게 구현되나?

두 가지 레벨이 있어.

### 레벨 1: 암묵적 반복 (기본, 추가 코드 거의 불필요)

`create_agent`로 만든 에이전트는 애초에 **ReAct 루프 자체가 반복 구조**야. Retriever가 그냥 하나의 도구니까:

```
LLM: "search_docs 도구로 검색해야겠다" → 검색 실행 → 결과 별로임
LLM: (결과가 부실하다고 판단) "다른 키워드로 search_docs 다시 호출해야겠다" → 재검색
LLM: "이제 충분한 정보 얻었다" → 최종 답변
```

이건 **별도 코드 없이 저절로 일어남** — LLM이 도구 결과를 보고 "이걸로 답할 수 있나?"를 매 턴마다 스스로 판단하기 때문. 개발자가 할 일은 `max_iterations`(또는 LangGraph의 recursion limit) 정도로 무한 루프만 막아주면 됨.

### 레벨 2: 명시적 반복 제어 (Corrective RAG 같은 고급 패턴)

"검색 결과가 관련 있는지 **별도로 평가**하고, 점수 낮으면 강제로 쿼리 재작성 후 재검색"처럼 로직을 명시적으로 그래프에 그리고 싶으면, `create_agent`가 아니라 **LangGraph의 `StateGraph`를 직접 구성**해야 해.

```python
from langgraph.graph import StateGraph, END

def retrieve(state):
    docs = retriever.invoke(state["query"])
    return {"docs": docs}

def grade(state):
    # LLM으로 "이 문서들이 질문과 관련 있나?" 평가
    is_relevant = grade_documents(state["docs"], state["query"])
    return {"is_relevant": is_relevant}

def rewrite_query(state):
    new_query = rewrite(state["query"])
    return {"query": new_query}

def generate(state):
    answer = generate_answer(state["docs"], state["query"])
    return {"answer": answer}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate", generate)

graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    lambda state: "generate" if state["is_relevant"] else "rewrite_query",
)
graph.add_edge("rewrite_query", "retrieve")  # ← 여기가 "반복"이 실제로 일어나는 지점
graph.add_edge("generate", END)
```

**정리:** "필요하면 반복" 중 단순한 경우(재검색 여부 판단)는 `create_agent`가 알아서 해주고, "관련성 점수 매기고 강제로 재시도시키는" 정교한 제어가 필요할 때만 `StateGraph`로 직접 루프(`add_edge`로 다시 앞 노드로 되돌리기)를 그리면 돼. 지금 단계에선 레벨 1만 알아도 충분하고, 레벨 2는 나중에 검색 품질이 실제 문제가 될 때 파면 됨.

---

## 7. 마지막 실습들, LangGraph/`create_agent`로 하는 게 나은가?

**결론: 그렇게 하는 걸 추천.** 이유:

- 교재 코드(`AgentExecutor` 기반)를 그대로 따라 치면 "동작은 하지만 지금 실무에서 안 쓰는 API"를 익히는 셈이 됨
- 반면 개념(Tool 정의, description 작성, 순환 구조 이해)은 어차피 두 방식 다 100% 동일하게 적용되니까, **실습만 `create_agent` 기준으로 바꿔서 짜는 게 시간 대비 이득이 큼** — 코드량도 오히려 줄어듦

**추천 순서:**
1. 교재 예제를 먼저 `AgentExecutor`로 한 번 돌려보고 "이게 왜 이렇게 동작하는지" 개념 확인 (지금까지 한 것처럼)
2. 그 다음 똑같은 예제를 `create_agent`로 다시 짜보기 — 코드가 얼마나 간결해지는지, HITL/메모리 설정이 어떻게 달라지는지 직접 비교
3. CSV 분석 에이전트, 파일 관리 에이전트처럼 "완성형 헬퍼 함수"(`create_pandas_dataframe_agent`, `FileManagementToolkit`) 쓰는 것들은 굳이 안 바꿔도 됨 — 이런 건 도구 자체가 여전히 유효하고, `create_agent`와 조합해서 쓰면 됨

한화 해커톤/AX 전환 준비 목적이면, 포트폴리오에 올라갈 최종 프로젝트는 `create_agent` 기준으로 짜는 게 "최신 스택 다룰 줄 안다"는 신호로 더 잘 먹힐 거야.
