# 05. 실전 에이전트 — 데이터 분석 / 파일 관리

## CSV/Excel 데이터 분석 에이전트

`create_pandas_dataframe_agent`로 pandas DataFrame을 자연어로 조작하는 Agent를 한 번에 생성.

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

df = pd.read_csv("data.csv")

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    df=df,
    verbose=True,
    allow_dangerous_code=True,  # 내부적으로 코드를 실행하므로 필수
)

agent.invoke("이 데이터에서 매출이 가장 높은 상위 5개 제품은?")
```

**내부 동작**: LLM이 질문을 보고 pandas 코드를 직접 생성 → `PythonREPLTool` 계열 도구로 실행 → 결과를 자연어로 정리해서 반환. 즉 앞서 배운 "LLM이 코드 작성 → Tool로 실행 → 결과 피드백" 루프를 pandas 전용으로 미리 만들어둔 패키지다.

### `df.corr()`이 뭔지

pandas의 `DataFrame.corr()`은 **모든 숫자형 컬럼 쌍 사이의 상관계수(correlation coefficient)를 계산**하는 메서드다.

```python
df.corr()
#           매출     광고비    재고량
# 매출      1.00    0.85    -0.12
# 광고비    0.85    1.00    -0.05
# 재고량   -0.12   -0.05     1.00
```

- 값의 범위는 -1 ~ 1
- **1에 가까움**: 두 변수가 강한 양의 상관관계 (하나가 오르면 다른 하나도 오름)
- **-1에 가까움**: 강한 음의 상관관계 (하나가 오르면 다른 하나는 내려감)
- **0에 가까움**: 뚜렷한 선형 관계 없음
- 데이터 분석에서 "어떤 변수가 매출에 영향을 주는지" 같은 질문에 기초적으로 쓰이는 함수. Agent가 "매출과 관련 있는 요인이 뭐야?"라는 질문을 받으면 이 함수를 스스로 호출한 것.
- 주의: 상관관계는 **인과관계가 아님**. 상관계수가 높다고 해서 한쪽이 다른 쪽의 원인이라고 단정할 수 없음.

### 주의사항

- `allow_dangerous_code=True`가 필수인 이유 그대로 **임의 Python 코드를 실행**함 → 신뢰 안 되는 입력이 들어오면 보안 리스크. 사용자가 업로드한 파일을 그대로 처리하는 서비스라면 샌드박스 격리 필수
- `langchain_experimental`에 있다는 것 자체가 "아직 실험적"이라는 신호. API가 버전마다 바뀔 수 있어서, 개념 학습용으로는 좋지만 프로덕션에 그대로 박아넣기보다는 동작 원리를 이해하는 용도로 보는 게 안전

## 파일 관리 자동화 에이전트

`FileManagementToolkit`으로 파일 읽기/쓰기/복사/삭제 등을 도구 묶음(toolkit)으로 한 번에 제공.

```python
from langchain_community.agent_toolkits import FileManagementToolkit

toolkit = FileManagementToolkit(root_dir="./workspace")
tools = toolkit.get_tools()
# 기본 제공: CopyFileTool, DeleteFileTool, FileSearchTool,
#            MoveFileTool, ReadFileTool, WriteFileTool, ListDirectoryTool 등

agent = create_tool_calling_agent(llm, tools, prompt)
```

**핵심 포인트**: `root_dir`을 지정하면 그 디렉터리 밖으로 나가지 못하도록 제한할 수 있음 — Agent가 시스템 전체 파일을 건드리는 걸 막는 최소한의 안전장치. 실무에서 파일 관리 Agent를 만들 때는 이 sandbox 제한이 사실상 필수.

## 나머지 실습들 (사용자 정의 함수 기반)

교재 뒷부분 실습들이 "사용자 정의 함수 만들어서 처리"하는 느낌이었다고 했는데, 이건 자연스러운 흐름이다. 빌트인 도구(Tavily, DALL·E, FileManagementToolkit 등)로 커버 안 되는 업무 로직(보고서 포맷팅, 특정 사내 API 호출, 도메인 특화 계산 등)은 결국 `@tool` 데코레이터로 직접 함수를 짜서 Agent에 쥐여주는 수밖에 없다. 지금까지 배운 개념(도구 정의 → 바인딩 → Agent가 판단 → 실행)이 결국 실무에서도 그대로 반복되는 패턴이라는 걸 확인하는 파트로 보면 된다.

## 2026년 기준 업데이트

- `create_pandas_dataframe_agent`, `FileManagementToolkit` 자체는 여전히 쓸 수 있지만, 이들을 **`AgentExecutor` 대신 `create_agent`(LangGraph 기반)로 감싸는 조합**이 현재 더 권장됨
- 실무형 "CSV 분석 에이전트"를 제대로 만들려면, 단순 pandas 코드 생성뿐 아니라 **코드 실행 결과 검증(생성된 코드가 실제로 맞는 답을 냈는지 확인하는 단계)**을 추가하는 경우가 많음 — 지금은 개념만 알아두고, 나중에 정확도가 중요한 프로젝트를 할 때 고려하면 됨
