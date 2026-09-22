# 04. Agentic RAG

## 일반 RAG vs Agentic RAG

**일반 RAG**: 고정 파이프라인. 질문 → 무조건 검색 → 검색 결과로 답변 생성.
검색이 필요 없는 질문에도 검색하고, 검색 결과가 부실해도 그대로 답변을 만든다.

**Agentic RAG**: 검색 자체를 Agent가 판단해서 호출하는 **도구(Tool) 중 하나**로 취급.

1. 검색할지 말지부터 판단 ("이 질문은 문서 안 뒤져도 답할 수 있네" → 스킵 가능)
2. 여러 검색 전략(도구) 중 선택 — 문서 검색(벡터 DB) vs 웹 검색(Tavily) 등
3. 검색 결과 품질을 스스로 평가 → 부실하면 쿼리 재작성 후 재검색
4. 필요하면 여러 번 반복(multi-hop)

## 구현 흐름

```python
from langchain.tools.retriever import create_retriever_tool

# 1. Retriever를 Tool로 감싸기
retriever_tool = create_retriever_tool(
    retriever,
    name="search_internal_docs",
    description="사내 문서에서 정보를 찾을 때 사용. 정책, 매뉴얼 관련 질문에 적합.",
)

# 2. 웹 검색 도구도 준비 (예: Tavily)
web_search_tool = TavilySearch(max_results=3)

# 3. 두 도구를 함께 바인딩
tools = [retriever_tool, web_search_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

## 도구 선택 우선순위는 어떻게 결정되나

"어떤 걸 우선순위로 하는지"에 대한 답: **LLM이 각 도구의 `description`만 보고 스스로 판단**한다. 별도의 우선순위 랭킹 로직이 코드로 정해져 있는 게 아니다.

그래서 실무에서 우선순위를 조정하는 방법은 코드가 아니라 **프롬프트/description 설계**로 이뤄진다:

| 방법 | 예시 |
|---|---|
| description에 우선순위 명시 | "사내 정책 질문은 반드시 `search_internal_docs`를 먼저 사용하고, 그래도 답이 없으면 웹 검색을 사용하세요" |
| 시스템 프롬프트에 전략 명시 | "너는 먼저 내부 문서를 확인하고, 최신 정보(오늘 날짜, 실시간 데이터)가 필요할 때만 웹 검색을 사용하는 어시스턴트다" |
| 도구 이름 자체를 명확히 | `search_internal_docs` vs `search_company_policy_docs`처럼 구체적으로 쓸수록 LLM이 헷갈릴 확률이 줄어듦 |

즉, description을 얼마나 명확하게 쓰느냐가 사실상 Agentic RAG 품질의 8할을 좌우한다. 이게 앞서 얘기했던 "description 튜닝이 은근 노가다"인 이유.

## 검색 품질 평가 (고급 개념 — 참고용)

교재보다 한 단계 더 나가면, 검색 결과가 질문과 관련 있는지 **별도의 평가(grading) 단계**를 두는 패턴도 있다 (예: Corrective RAG, Self-RAG 계열 논문에서 나온 아이디어). LangGraph로 구현할 때는 조건 분기(conditional edge)로 "검색 결과 관련성 낮음 → 쿼리 재작성 → 재검색" 루프를 명시적으로 그래프에 그릴 수 있다. 지금 단계에서는 개념만 알아두면 충분하고, 실제 구현은 나중에 필요할 때 봐도 됨.
