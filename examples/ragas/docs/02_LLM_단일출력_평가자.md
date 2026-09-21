# 02. LLM 기반 단일 출력 평가자 — LLM-as-a-Judge와 그 종류들

> 전체 지도: [00_전체_흐름과_빌드업.md](00_전체_흐름과_빌드업.md)
> 대조되는 내용: [03_휴리스틱_평가지표.md](03_휴리스틱_평가지표.md) (LLM 안 쓰는 채점 방식)
> 실전 운영: [04_랭스미스_운영_평가기능.md](04_랭스미스_운영_평가기능.md) (여기서 배운 평가자를 실제로 돌리는 방법)

**챕터 전환 알림**: 여기서부터 교재는 "랭스미스 API를 활용한 프롬프트 최적화"라는 새 챕터로 넘어가. 이 챕터 전체의 목적은 — **평가 점수를 근거로 프롬프트/파이프라인을 개선(최적화)하는 것**. 그러려면 먼저 "점수를 어떻게 매기는지"부터 알아야 해서, 이 파일에서는 채점 방법(어휘/도구)을 배우는 단계야.

---

## 평가 데이터셋 구축의 3가지 관계 축 (핵심 개념 — 가장 중요)

RAG 시스템에는 세 가지 요소가 있어: **질문(Question) - 검색된 문서(Retrieval/Context) - 답변(Answer)**. 이 셋을 꼭짓점으로 하는 삼각형을 그리면, 평가자는 결국 이 삼각형의 **변(edge) 하나씩을 채점**하는 거야.

```
            Question (질문)
             /          \
            /            \
   ① Q-Retrieval    ② Q-Answer
      (관련성 있나?)    (질문에 맞게 답했나?)
          /                \
   Retrieval ------------- Answer
       (검색된 문서)   ③ Retrieval-Answer    (생성된 답변)
                    (답변이 문서에 근거했나?)
```

| 관계 | 확인하는 것 | 대응하는 평가자 |
|---|---|---|
| **① Question-Retrieval** | 검색된 문서가 질문과 관련 있는가 | Relevance Grader (`target="retrieval-question"`) |
| **② Question-Answer** | 답변이 질문에 제대로 부합하는가 (정답과 비교) | QA 평가자 |
| **③ Retrieval-Answer** | 답변이 검색된 문서에 근거했는가 (환각 여부) | Context-QA 평가자, Groundedness 평가자 |

**이 삼각형이 이번 파트 전체를 관통하는 뼈대**야. 뒤에 나오는 평가자들(QA, Context-QA, Groundedness 등)이 각각 이 세 변 중 어디에 해당하는지 알면, "이게 왜 필요한 평가자인지"가 명확해져. [04번 파일](04_랭스미스_운영_평가기능.md)의 Groundedness 평가자도 결국 ③번 축이야.

---

## LLM-as-a-Judge (핵심 개념)

말 그대로 **"LLM을 심사위원으로 쓴다"**는 방법론. 사람이 채점하거나 문자열을 기계적으로 비교하는 대신, **또 다른 LLM에게 "이거 평가해줘"라고 시켜서 그 LLM이 점수를 매기게** 하는 방식이야.

```
정답: "대한민국의 수도는 서울입니다."
모델 답: "서울이 한국의 수도예요."
```

이 둘은 의미는 같지만 글자는 다르지. 단순 문자열 비교로는 못 잡아내는 이런 **의미적 일치**를 판단하려고 LLM을 심사위원으로 쓰는 거야.

**중요**: 이건 RAGAS가 발명한 방법이 아니라, LLM 평가 전반에 쓰이는 **일반적인 원리**야. RAGAS도 쓰고, 지금 배우는 LangChain의 평가자들도 이 원리 위에서 동작해.

---

## Off-the-shelf 평가자 (핵심 개념 — 상위 카테고리)

LangChain이 "자주 쓰이는 평가 유형별로 미리 완성된 평가 체인 클래스"를 제공하는 것. 이름만 지정하면 바로 갖다 쓸 수 있어.

```python
from langchain.evaluation import load_evaluator
evaluator = load_evaluator("qa")
```

**"LLM 평가자"라는 표현은 이 전체 카테고리를 가리키는 상위 분류명**이야. 아래 나오는 QA/Context-QA/Criteria/임베딩거리 평가자가 전부 이 "off-the-shelf 평가자"의 구체적인 종류(하위 항목)들이고, 서로 **같은 레벨**에 나열되는 선택지들이야.

---

## 평가자 종류별 정리 (같은 레벨의 4가지)

### 1. Question-Answer 평가자 (`QAEvalChain`)
**정답(ground truth)이 있을 때**, 답변이 정답과 얼마나 일치하는지 LLM이 판단. → 위 삼각형의 **②번(Q-Answer)** 축.

```python
evaluator = load_evaluator("qa")
evaluator.evaluate_strings(input="질문", prediction="모델답", reference="정답")
```

### 2. Context-Answer 평가자 (`ContextQAEvalChain`)
**정답이 없어도**, 주어진 컨텍스트만 보고 답변이 타당한지 판단. → 위 삼각형의 **③번(Retrieval-Answer)** 축. RAG의 환각(hallucination) 탐지 발상과 직결됨.

```python
evaluator = load_evaluator("context_qa")
evaluator.evaluate_strings(input="질문", prediction="모델답", reference="컨텍스트")
```

### 3. Criteria 평가자 (`CriteriaEvalChain`)
정답도 컨텍스트도 없이, **임의의 기준**을 만족하는지 판단. 가장 유연함.

```python
evaluator = load_evaluator("criteria", criteria="conciseness")
```

기본 제공 기준: `conciseness`(간결성), `relevance`(관련성), `correctness`(정확성), `coherence`(일관성), `harmfulness`(유해성), `maliciousness`(악의성), `helpfulness`(도움됨), `controversiality`(논쟁성), `misogyny`(여성혐오), `criminality`(범죄성). 커스텀 기준도 직접 정의 가능:

```python
custom_criteria = {"korean-only": "답변이 한국어로만 작성되었는가?"}
```

> 이 기준 목록 자체는 **부가 디테일**(암기 불필요) — "이런 식으로 자유롭게 기준을 정할 수 있다"는 유연성만 기억하면 충분.

### 4. 임베딩 거리 기반 평가자 (`EmbeddingDistanceEvalChain`)
**LLM-as-a-Judge가 아님** — 답변과 정답을 각각 임베딩해서 벡터 거리(코사인 유사도 등)로 유사도 계산. LLM 호출 없이 빠르고 저렴함.

```python
evaluator = load_evaluator("embedding_distance")
```

---

## 사용자 정의 평가자 (부가 디테일 — 자연스러운 확장)

위 4가지 off-the-shelf 평가자로 안 맞는 경우, **직접 프롬프트를 짜서 자기만의 채점 기준을 만드는 것**. 새로운 카테고리라기보다는, "미리 만들어진 것 vs 내가 직접 만든 것"의 연장선이라고 보면 돼. Criteria 평가자에서 커스텀 기준을 정의하는 것의 더 자유로운 버전.

---

## 실무 사용 여부

| 항목 | 실무 사용 |
|---|---|
| LLM-as-a-Judge (개념) | **매우 자주 씀** — RAGAS를 포함한 대부분의 RAG 평가 도구의 근본 원리 |
| LangChain off-the-shelf 평가자 (QA/Context-QA/Criteria/임베딩거리) | **RAG 평가에는 잘 안 씀**. RAGAS 같은 RAG 특화 도구가 이 영역을 대체함 |
| 같은 평가자들, RAG 아닌 용도 | 챗봇 톤 체크(Criteria), 일반 QA봇 정답 체크(QA), 요약 품질 체크(Criteria/Context-QA 응용) 등에는 여전히 쓰임 |
| 전체 트렌드 | 평가 인프라 무게중심이 `langchain.evaluation` → **LangSmith `evaluate()`**로 이동 중. 평가 로직은 자유롭게 고르되(RAGAS든 커스텀 함수든), 실행/기록은 LangSmith가 담당하는 조합이 표준이 되어가는 추세 |

**왜 RAG 평가에선 잘 안 쓰이나**: 이 평가자들은 범용으로 설계돼서, "리트리버가 문제인지 생성기가 문제인지" 같은 RAG 고유의 진단을 못 해줘. RAG 특화 진단이 필요하면 자연스럽게 RAGAS([05번 파일](05_RAGAS_RAG_파이프라인_평가.md))로 가게 됨.
