# 02. Tool Binding → Tool Calling Agent → AgentExecutor

## Tool Binding

LLM에게 "이런 도구들을 쓸 수 있어"라고 스키마(이름, 설명, 파라미터)를 등록하는 과정.

```python
llm_with_tools = llm.bind_tools([tool1, tool2])
```

- 이 시점에는 아무것도 실행되지 않는다. LLM이 도구 존재를 "인식"하게만 만드는 단계.
- 문제는 도구가 많아지고 로직이 복잡해질수록, `bind_tools` + 수동 실행 루프를 직접 짜는 게 번거로워진다.
  → 그래서 교재는 **Tool Calling Agent**라는 더 높은 추상화 단계로 넘어간다.

## Tool Calling Agent와 ReAct 순환

Tool Calling Agent의 핵심은 **순환(loop) 구조**다. 교재에서 이 흐름을 **Thought → Action → Observation** 개념으로 설명한다 (ReAct 패턴).

```
Thought (LLM이 "무엇을 해야 하나" 추론)
   ↓
Action (도구 호출 — tool_calls 생성, 이건 "요청"일 뿐 실행 아님)
   ↓
Observation (실제 도구 실행 결과를 LLM에게 다시 전달)
   ↓
다시 Thought로 돌아가서 반복, 혹은 "이제 최종 답변 가능" 판단되면 종료
```

**중요한 구분:**
- LLM은 "이 도구를 이 인자로 불러야겠다"는 **판단(tool_calls)만** 만든다.
- 실제 실행은 개발자 코드 또는 실행기(Executor)가 담당한다.
- 실행 결과(Observation)를 다시 LLM에게 넘겨야 다음 Thought가 만들어진다.

## AgentExecutor

이 Thought-Action-Observation 루프를 **자동으로 반복 실행**해주는 실행기.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # 각 단계 로그 출력
    max_iterations=15,  # 무한 루프 방지 (기본값)
)

result = agent_executor.invoke({"input": "서울 날씨 알려줘"})
```

### 주요 옵션

| 옵션 | 역할 |
|---|---|
| `verbose` | Action/Observation 각 단계를 콘솔에 출력 |
| `max_iterations` | 순환 횟수 제한 (무한 루프 방지) |
| `handle_parsing_errors` | LLM 출력 파싱 실패 시 에러 대신 재시도하도록 처리 |
| `return_intermediate_steps` | 최종 답변뿐 아니라 중간 Action/Observation 기록도 반환 |

### 중간 단계 스트리밍

`agent_executor.stream(...)`으로 Action/Observation이 실시간으로 어떻게 진행되는지 확인 가능. 디버깅과 사용자 경험(로딩 상태 표시) 둘 다에 유용.

### `.iter()` — 단계별 실행 확인

`agent_executor.iter(...)`를 쓰면 한 스텝씩 직접 순회하면서, 각 Action이 실행되기 **직전에 개입**할 수 있다. 이게 바로 다음 개념인 Human-in-the-Loop(HITL)의 기반.

### Human-in-the-Loop (HITL)

Agent가 특정 도구(특히 파괴적이거나 비용이 큰 작업 — 이메일 발송, 결제, 파일 삭제 등)를 실행하기 전에 **사람의 승인을 받도록** 중간에 끼워 넣는 패턴.

```python
for step in agent_executor.iter({"input": "..."}):
    if "intermediate_step" in step:
        action, observation = step["intermediate_step"]
        if action.tool == "send_email":
            confirm = input(f"{action.tool_input} 실행할까요? (y/n): ")
            if confirm != "y":
                break  # 실행 중단
```

## 2026년 기준 업데이트

**`AgentExecutor`는 현재 레거시(유지보수 모드) 상태다.** 보안 패치만 들어가고 신기능은 없다. 새 프로젝트는 다음을 쓰는 게 권장된다:

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=tools)

# 스트리밍
for chunk in agent.stream({"messages": [{"role": "user", "content": "..."}]}):
    print(chunk)
```

- `create_agent`는 내부적으로 **LangGraph의 StateGraph** 위에서 동작한다.
- ReAct 패턴(Thought-Action-Observation)을 내부에 이미 구현하고 있어서 코드량이 크게 줄어든다.
- **Human-in-the-Loop도 `iter()`로 수동 구현하지 않고, LangGraph의 미들웨어/인터럽트(interrupt) 기능으로 표준화**되어 있다 — 특정 도구 실행 전에 그래프 실행을 자동으로 멈추고 사람 승인을 기다리는 방식.
- `max_iterations` 같은 옵션도 `create_agent`/LangGraph 쪽 파라미터로 대응된다.

**정리:** 개념(ReAct 순환, Action/Observation, HITL)은 그대로 배우되, 실제 구현 코드는 `AgentExecutor.iter()` 수동 패턴보다 `create_agent` + LangGraph의 내장 interrupt 방식이 현재 표준이라는 것만 알아두면 된다.
