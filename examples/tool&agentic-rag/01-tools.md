# 01. 도구(Tools) 기초

## 도구란

도구(Tool)는 에이전트, 체인, LLM이 **외부 세계와 상호작용하기 위한 인터페이스**다.
LLM 자체는 텍스트만 생성할 수 있으므로, 실제 계산·검색·파일 조작 등을 하려면 도구가 필요하다.

## 빌트인 도구 (Built-in Tools)

교재에서 다룬 대표 도구들:

| 도구 | 역할 | 비고 |
|---|---|---|
| `PythonREPLTool` | Python 코드를 실행하는 REPL 환경 제공 | `langchain_experimental` 소속, 외부 API 키 불필요 |
| Tavily Search | LLM 에이전트 특화 웹 검색 | 외부 서비스(Tavily), API 키 필요, 패키지는 `langchain-tavily`로 분리됨 |
| DALL·E 이미지 생성 (`DallEAPIWrapper`) | 텍스트 → 이미지 생성 | OpenAI API 키 필요 |

### 핵심 구분: 코어 내장 vs 외부 통합

- **LangChain 코어 자체 기능**: `create_retriever_tool` 같은 헬퍼 — 외부 서비스 없이 동작
- **외부 서비스를 감싼 통합(Integration)**: Tavily Search, DALL·E 등 — 별도 API 키/패키지 필요

```python
from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()
print(python_tool.invoke("print(100 + 200)"))
```

> ⚠️ `PythonREPLTool`은 임의 코드를 실제로 실행하므로, 신뢰할 수 없는 입력을 그대로 넣으면 보안 위험이 있다. 프로덕션에서는 샌드박스 격리가 필수.

## 커스텀 도구 (Custom Tools)

`@tool` 데코레이터로 어떤 Python 함수든 Agent가 쓸 수 있는 도구로 만들 수 있다.

```python
from langchain_core.tools import tool

@tool
def get_word_length(word: str) -> int:
    """단어의 길이를 반환합니다."""
    return len(word)
```

**도구가 되려면 필요한 것:**
- **name**: 도구 이름 (함수명이 기본값)
- **description**: LLM이 "언제 이 도구를 써야 할지" 판단하는 근거 → **가장 중요한 부분**. description이 모호하면 Agent가 도구를 잘못 고르거나 안 써야 할 때 씀
- **args_schema**: 입력 파라미터 타입 (타입 힌트로 자동 추론되거나 Pydantic 모델로 명시)

### 2026년 기준 참고

- `@tool` 데코레이터, docstring 기반 description 작성 방식은 지금도 그대로 유효함 (가장 안정적인 부분)
- 최신 `langchain.tools`(`from langchain.tools import tool`)로 임포트 경로가 정리된 경우도 있으니, 실습 시 버전에 따라 `langchain_core.tools`인지 `langchain.tools`인지 확인 필요
