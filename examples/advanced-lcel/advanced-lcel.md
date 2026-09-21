# LCEL 고급 문법 정리

## 1. LCEL과 Runnable 기본 개념

### LCEL (LangChain Expression Language)
LangChain 컴포넌트들을 `|` (파이프) 연산자로 연결해서 체인을 만드는 선언적 문법.

```python
chain = prompt | model | output_parser
```

내부적으로는 `RunnableSequence(prompt, model, output_parser)`를 만드는 것이며, 앞 단계의 출력이 자동으로 다음 단계의 입력으로 흘러간다.

**장점**
- 코드가 짧고 가독성이 좋음
- 스트리밍/배치/비동기가 체인 전체에 자동 적용됨
- 체인 중간 단계를 쉽게 붙이고 뗄 수 있음

### Runnable
LCEL로 연결 가능한 **모든 구성요소의 공통 인터페이스**. `prompt`, `model`, `retriever`, `output_parser` 모두 이 인터페이스를 구현하고 있어서 `|`로 연결 가능하다.

| 메서드 | 설명 |
|---|---|
| `invoke(input)` | 단일 입력 → 단일 출력 |
| `batch(inputs)` | 여러 입력 → 병렬 처리 |
| `stream(input)` | 토큰 단위 스트리밍 |
| `ainvoke` / `abatch` / `astream` | 각각의 비동기 버전 |

체인 자체도 하나의 Runnable이기 때문에, **체인 안에 체인을 중첩**시켜 재귀적으로 합성할 수 있다.

---

## 2. RunnablePassthrough / RunnableParallel

### RunnablePassthrough
입력을 **그대로 다음 단계로 전달**. 원본 값을 유지해야 할 때 사용.

```python
from langchain_core.runnables import RunnablePassthrough

RunnablePassthrough().invoke("안녕")  # -> "안녕" (그대로 반환)
```

### RunnableParallel
여러 Runnable을 **동시에 실행**해서 결과를 하나의 딕셔너리로 묶어줌.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)
# {"context": 검색결과, "question": 원본질문}
```

**딕셔너리 리터럴을 체인에 넣으면 자동으로 RunnableParallel로 변환된다.**

```python
# 아래 둘은 완전히 동일
chain = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = {"context": retriever, "question": RunnablePassthrough()}
```

#### 핵심 용도: 데이터 형태(shape) 변환 어댑터
앞 단계의 출력 형식과 다음 단계가 기대하는 입력 형식이 다를 때, 여러 소스의 값을 모아 다음 단계가 원하는 딕셔너리 구조로 **재조립**하는 역할을 한다.

```python
# retriever는 문서 리스트를 반환하지만, prompt는 {context, question} 딕셔너리를 기대함
# → RunnableParallel이 그 간극을 메꿔줌
{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt
```

### RunnablePassthrough.assign()
원본 딕셔너리를 **유지하면서 새 키를 추가**(또는 같은 키를 재계산해서 덮어쓰기)하는 메서드.

```python
chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["question"].split())
)
chain.invoke({"question": "안녕 반가워"})
# -> {"question": "안녕 반가워", "word_count": 2}
```

**키가 겹치면 덮어씀(overwrite).** 이 성질을 활용해 `context`를 문서 리스트 → 문자열로 변환하며 같은 키에 재할당하는 패턴이 자주 쓰인다.

```python
chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(doc.page_content for doc in x["context"])
      )
)
```

### 세 가지 비교 (공식 예제)

```python
runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)
runnable.invoke({"num": 1})
# {'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
```

| 키 | 방식 | 원본 유지? | 결과 |
|---|---|---|---|
| `passed` | `RunnablePassthrough()` | 그대로 | `{'num': 1}` |
| `extra` | `.assign(...)` | 유지 + 추가 | `{'num': 1, 'mult': 3}` |
| `modified` | 일반 함수 (자동으로 RunnableLambda) | 유지 안 함 | `2` |

> `passed`, `extra`, `modified`는 문법 용어가 아니라 **그냥 딕셔너리 키 이름**이다. `context`, `question`도 마찬가지로 자유롭게 지을 수 있는 키일 뿐이며, 실제로 맞춰야 하는 건 **다음 단계(프롬프트)의 `{변수명}`**이다.

---

## 3. 중첩 구조 이해하기

```python
{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
```

이 코드는 두 레벨의 `|`가 쓰인 **체인 안의 체인** 구조다.

```
retrieval_chain (RunnableSequence)
├── RunnableParallel
│   ├── "context" → RunnableSequence (retriever | format_docs)  ← 안쪽 체인
│   └── "question" → RunnablePassthrough()
├── prompt
├── model
└── StrOutputParser()
```

`retriever | format_docs`처럼 여러 Runnable을 이어붙인 것도 그 자체로 하나의 Runnable이기 때문에, `RunnableParallel`의 값 자리에 레고 블록처럼 끼워 넣을 수 있다. 이것이 LCEL이 "합성 가능하다(composable)"고 불리는 이유다.

---

## 4. 자동 변환(Coercion) 규칙

LCEL은 체인 안에 Runnable이 아닌 파이썬 객체가 들어오면 자동으로 알맞은 Runnable로 감싸준다.

| 체인 안에 넣은 것 | 자동 변환 |
|---|---|
| 딕셔너리 `{...}` | `RunnableParallel` |
| 일반 함수/콜러블 (`def`, `lambda`) | `RunnableLambda` |

```python
retriever | format_docs
# 내부적으로 → retriever | RunnableLambda(format_docs)
```

---

## 5. 완성형 RAG 체인 예제

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts([...], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
)
model = ChatOpenAI(model_name="gpt-4o-mini")

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

**흐름**: 질문 → (검색해서 문맥 만들기 + 질문 원본 보존) → 프롬프트에 둘 다 채우기 → LLM 호출 → 문자열로 파싱

---

## 6. 라우팅 (Routing)

질문 종류에 따라 다른 체인으로 분기하는 패턴. 두 가지 방법이 있으며, **공식 문서는 커스텀 함수(RunnableLambda) 방식을 권장**한다 (RunnableBranch는 레거시).

### 방법 1: RunnableLambda + 커스텀 함수 (권장)

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

def route(info):
    if "수학" in info["topic"].lower():
        return math_chain
    elif "과학" in info["topic"].lower():
        return science_chain
    else:
        return general_chain

full_chain = (
    {"topic": classification_chain, "question": itemgetter("question")}
    | RunnableLambda(route)
    | StrOutputParser()
)
```

**핵심 원리**: `route` 함수는 값이 아니라 **다음에 실행할 체인(Runnable) 자체를 반환**한다. `RunnableLambda`가 반환한 값이 Runnable이면, LCEL은 그걸 곧바로 실행해서 이어붙인다.

### 방법 2: RunnableBranch (레거시)

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "수학" in x["topic"].lower(), math_chain),
    (lambda x: "과학" in x["topic"].lower(), science_chain),
    general_chain,  # default
)
```

`(조건함수, 체인)` 튜플을 순서대로 검사해서 처음 참이 되는 조건의 체인을 실행. 다 거짓이면 마지막 default 실행.

### 두 방식 비교

| | RunnableLambda + 함수 | RunnableBranch |
|---|---|---|
| 상태 | 권장 | 레거시 |
| 자유도 | 높음 (아무 로직이나 가능) | 단순 조건-체인 쌍에 최적화 |
| 가독성 | 조건 복잡할 때 유리 | 조건 단순할 때 한눈에 보기 좋음 |

---

## 7. RunnableConfig

### 개념
모든 Runnable의 `invoke()` / `batch()` / `stream()`이 공통으로 받는 **실행 시점 부가 설정**. 체인이 처리하는 실제 데이터(input)와는 별개로, 각 단계를 거치며 변하지 않고 그대로 전파된다.

```python
chain.invoke(
    "질문",                                  # ← input (실제 데이터, 단계마다 변함)
    config={"tags": ["test-run"], "max_concurrency": 3}  # ← config (메타 설정, 안 변함)
)
```

담을 수 있는 값: `callbacks`, `tags`, `metadata`, `max_concurrency`, `run_name`, `configurable` 등. 주로 LangSmith 같은 트레이싱 툴에서 실행을 추적/분류하는 데 쓰인다.

### 식별자 여부 정리

| | 성격 | 이름 바꿔도 되나? |
|---|---|---|
| `config` | `invoke()`의 **파라미터 이름** (고정) | 불가능 |
| `configurable` | `RunnableConfig` 스키마의 **고정 필드명** | 불가능 |
| `configurable` 안의 키 (예: `gpt_version`) | `ConfigurableField(id=...)`로 직접 지은 이름 | 자유 |

---

## 8. configurable_fields — 필드값 단위 교체

컴포넌트의 **특정 필드 값 하나**를 실행 시점에 바꿀 수 있게 노출.

```python
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(temperature=0).configurable_fields(
    model_name=ConfigurableField(
        id="gpt_version",                       # 실행 시 config에서 참조하는 식별자 (필수)
        name="Version of GPT",                   # 사람이 보는 표시용 이름 (문서화/UI용)
        description="Official model name of GPTs. ex) gpt-4o, gpt-4o-mini",  # 설명 (문서화/UI용)
    )
)

model.invoke(
    "질문",
    config={"configurable": {"gpt_version": "gpt-3.5-turbo"}},
)
```

> `id`만 실행 로직에 쓰이고, `name`/`description`은 LangServe Playground 같은 자동 생성 UI에서 라벨/도움말로 노출되는 **문서화용 메타데이터**다.

---

## 9. configurable_alternatives — 컴포넌트 통째로 교체

`configurable_fields`가 "필드 값 하나"를 바꾸는 거라면, `configurable_alternatives`는 **미리 준비해둔 완성된 인스턴스 전체**를 실행 시점에 통째로 갈아끼우는 것.

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

model = ChatAnthropic(model="claude-3-sonnet-20240229").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="anthropic",
    openai=ChatOpenAI(model="gpt-4o"),
    gpt4=ChatOpenAI(model="gpt-4-turbo"),
)

chain = prompt | model | StrOutputParser()

chain.invoke({"question": "..."})  # 기본값: ChatAnthropic 사용
chain.invoke({"question": "..."}, config={"configurable": {"llm": "openai"}})  # 통째로 교체
```

**주의**: "클래스가 다르냐"가 기준이 아니라 **"바꾸는 단위가 필드 하나냐, 여러 설정이 조합된 완성 객체 전체냐"**가 진짜 기준이다. 같은 클래스여도(예: `VectorStoreRetriever` 두 개) 여러 파라미터가 세트로 묶여야 하면 alternatives가 더 적합하다.

### 실무 활용 사례

| 사례 | 설명 |
|---|---|
| 모델 공급사 교체 | OpenAI ↔ Anthropic ↔ 등 |
| 리트리버 검색 전략 | similarity ↔ MMR ↔ score_threshold |
| 프롬프트 톤/스타일 | formal ↔ casual ↔ concise |
| 임베딩 모델 크기 | small(저비용) ↔ large(고정확도) |
| A/B 테스트 / 카나리 배포 | `default_key`는 검증된 버전, 실험군만 새 버전 지정 |

### CSS font-family와의 차이 (주의할 점)
`configurable_alternatives`는 CSS `font-family`의 **자동 fallback**(1순위 실패 시 자동으로 2순위 시도)과는 다르다. config에서 **명시적으로 지정해야만** 대안으로 바뀌며, 지정 안 하면 무조건 `default_key`가 실행되고 에러가 나도 자동으로 다음 대안으로 안 넘어간다.

> 자동 실패 대응(fallback)이 필요하면 `.with_fallbacks()`를 따로 써야 한다.
> ```python
> model = ChatOpenAI().with_fallbacks([ChatAnthropic()])
> # OpenAI 호출 실패 시 자동으로 Anthropic으로 재시도
> ```

---

## 10. .with_config — 설정을 미리 고정한 새 Runnable 만들기

매번 `invoke(config=...)`를 넘기는 대신, 설정이 **미리 적용된 새로운 Runnable 사본**을 만들어두는 메서드. 원본은 건드리지 않는다.

```python
gpt35_model = model.with_config(configurable={"gpt_version": "gpt-3.5-turbo"})
gpt35_model.invoke("질문")  # config 안 넘겨도 항상 gpt-3.5-turbo로 동작
```

### 왜 필요한가 — "애초에 잘 만들면 되지 않나?"
정적으로 값 하나만 쓸 거면 처음부터 생성자에 박아 넣는 게 맞다. `.with_config`가 진짜 필요한 상황은:

1. **같은 체인 정의를 여러 변형으로 재사용**할 때 (코드 한 곳에서 파생)
2. **설정값이 실행 시점(런타임)에만 결정될 때** — 사용자 등급, 요청별 조건 등 코드 작성 시점엔 알 수 없는 값
3. **서버로 배포되어 외부 요청마다 다르게 동작**해야 할 때 (코드 재배포 없이 파라미터만 변경)

```python
# 요청이 들어온 순간에만 알 수 있는 값을 반영하는 예
def handle_request(user_id, question):
    model_name = "gpt-4o" if get_user_tier(user_id) == "premium" else "gpt-3.5-turbo"
    return chain.with_config(configurable={"gpt_version": model_name}).invoke(question)
```

---

## 11. HubRunnable — LangChain Hub 프롬프트 원격 연동

**LangChain Hub**: 프롬프트를 저장/공유/버전관리하는 저장소 ("프롬프트용 npm").

```python
from langchain import hub
prompt = hub.pull("hwchase17/react")  # 공개 프롬프트 일회성으로 가져오기
```

**HubRunnable**: Hub의 프롬프트를 Runnable로 감싸서, `configurable_fields`로 커밋(버전) 식별자를 노출 → **실행 시점에 다른 Hub 프롬프트로 교체 가능**하게 만든 클래스.

```python
from langchain.runnables.hub import HubRunnable

prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
    owner_repo_commit=ConfigurableField(id="hub_commit", ...)
)
chain.invoke(input, config={"configurable": {"hub_commit": "rlm/rag-prompt-llama"}})
```

### 로컬 분기 방식과의 차이
- **로컬 분기(RunnableLambda/Branch)**: 프롬프트 내용 자체가 이미 코드에 다 작성되어 있고, 그중 하나를 조건으로 선택
- **HubRunnable**: 프롬프트 내용을 코드에 안 두고, 실행 시점에 **원격(Hub 서버)에서 끌어옴** → 프롬프트 수정이 코드 배포와 분리되고, 버전 히스토리/롤백 관리 가능

---

## 12. @chain 데코레이터

일반 함수를 `@chain`으로 감싸면 Runnable(체인)으로 만들 수 있다. **기능적으로는 `RunnableLambda`로 감싸는 것과 동일**하다.

```python
from langchain_core.runnables import chain

@chain
def custom_chain(text):
    prompt_val1 = prompt1.invoke({"topic": text})
    output1 = ChatOpenAI().invoke(prompt_val1)
    parsed_output1 = StrOutputParser().invoke(output1)
    chain2 = prompt2 | ChatOpenAI() | StrOutputParser()
    return chain2.invoke({"joke": parsed_output1})

custom_chain.invoke("bears")  # 이제 하나의 Runnable
```

### RunnableLambda와의 차이 — 관찰성(Observability)
`@chain`을 쓰면 함수 내부에서 호출한 Runnable들이 LangSmith 트레이스에서 **중첩된 자식 노드로 계층화**되어 기록된다. 단순 `RunnableLambda`는 이 계층 구조가 잘 안 잡힐 수 있다.

> **관찰성(Observability)**: 시스템 내부에서 무슨 일이 일어나는지 외부에서 들여다볼 수 있게 만드는 것. 단순히 "보기 좋게" 만드는 게 아니라, 비결정적인 LLM 체인에서 **어느 단계가 문제였는지 정확히 특정**하고, 단계별 비용/성능을 추적하고, 프로덕션 이상 상황을 진단하기 위한 필수 인프라다.

### 선택 기준

| 상황 | 선택 |
|---|---|
| 순수 값 변환 (포맷팅, 키 추출) — 내부에서 다른 Runnable 호출 없음 | `RunnableLambda` (또는 그냥 함수, 자동 변환) |
| 함수 내부에서 **여러 Runnable을 호출**하며 조합/오케스트레이션 | `@chain` |
| 디버깅/모니터링 시 이 로직 내부를 단계별로 추적할 필요가 있음 | `@chain` |

> 실무에서는 대부분의 체인을 `|` 파이프만으로 직접 조립하며, `@chain`은 파이프만으로 표현하기 애매한 조건 분기·예외 처리가 섞인 로직에 보조적으로 쓰인다.

---

## 용어: "식별자" vs "자유 이름" 총정리

| 이름 | 성격 | 근거 |
|---|---|---|
| `context`, `question`, `passed`, `extra`, `modified` (딕셔너리 키) | 자유 | 프롬프트의 `{변수명}`과만 맞추면 됨 |
| `config` | 고정 식별자 | `invoke()` 메서드의 파라미터명 |
| `configurable` | 고정 식별자 | `RunnableConfig` 스키마의 필드명 |
| `configurable` 안의 키 (예: `gpt_version`) | 자유 | `ConfigurableField(id=...)`로 직접 지정 |
