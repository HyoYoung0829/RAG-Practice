# 05. RAGAS로 RAG 파이프라인 평가하기 — 빌드업의 종착점

> 전체 지도: [00_전체_흐름과_빌드업.md](00_전체_흐름과_빌드업.md)
> 앞서 나온 RAGAS: [01_RAGAS_합성_데이터셋.md](01_RAGAS_합성_데이터셋.md) (여기와 평가 대상이 다름 — 아래에서 설명)
> 재료가 된 개념들: [02_LLM_단일출력_평가자.md](02_LLM_단일출력_평가자.md)의 질문-문서-답변 삼각형

---

## 01번 파일과 뭐가 다른가 (가장 먼저 짚어야 할 것)

같은 RAGAS 라이브러리인데, **평가 대상이 완전히 다름**.

| | [01번 파일](01_RAGAS_합성_데이터셋.md) | 이 파일 |
|---|---|---|
| 평가 대상 | LLM이 만든 **질문-정답 세트 자체** | 내 **RAG 파이프라인의 실제 출력** |
| 질문 | "이 시험 문제가 좋은 문제인가?" | "내 RAG가 이 시험을 잘 봤는가?" |
| 시점 | 데이터셋 생성 직후 | RAG 체인을 다 만들고, 그 체인을 실제로 돌려본 후 |

즉 01번은 **"시험지 검수"**, 여기는 **"실제 시험 채점"**이야.

---

## 전체 3단계 흐름 복습

```
① 합성 테스트 데이터셋 생성 (LLM 사용) — 01번 파일
   문서 → [LLM] → 질문 + 정답 컨텍스트 + 정답

② 내 RAG 파이프라인을 그 질문들에 실제로 돌려봄
   for q in testset["question"]:
       result = my_rag_chain.invoke(q)
       answers.append(result["answer"])
       retrieved_contexts.append(result["context"])

③ RAGAS로 "정답 vs 내 파이프라인의 실제 답안"을 채점  ← 이 파일
   from ragas import evaluate
   from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

   result = evaluate(
       dataset=my_dataset,  # question, answer, retrieved_context, ground_truth 포함
       metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
   )
```

비유하면:
- ① = 선생님(LLM)이 시험 문제지 + 모범답안 만듦
- ② = 학생(내 RAG 시스템)이 실제로 문제를 풀어봄
- ③ = 채점관(RAGAS, 또 다른 LLM)이 모범답안과 학생 답안을 비교해서 점수 매김

**RAGAS가 평가하는 건 "데이터셋"이 아니라 "내 시스템이 그 데이터셋에 낸 결과물"**이야. 데이터셋은 문제지 역할, RAGAS는 채점 역할, 그 사이에 반드시 "내 시스템이 실제로 시험을 치르는" ②단계가 껴 있어야 해.

---

## RAGAS 핵심 지표 (핵심 개념)

[02번 파일](02_LLM_단일출력_평가자.md)에서 배운 **질문-문서-답변 삼각형**이 여기서도 그대로 적용돼. RAGAS의 대표 지표들은 결국 그 삼각형의 변을 RAG 전용으로 특화 구현한 것들이야.

| RAGAS 지표 | 확인하는 것 | 삼각형에서의 위치 |
|---|---|---|
| **Faithfulness** (충실성) | 답변이 검색된 컨텍스트에 근거하는가 | ③ Retrieval-Answer 축 ([04번 파일](04_랭스미스_운영_평가기능.md)의 Groundedness와 사실상 동일 개념) |
| **Context Precision** | 검색된 문서 중 실제로 관련 있는 게 얼마나 되는가 | ① Question-Retrieval 축 |
| **Context Recall** | 정답에 필요한 정보를 검색이 빠짐없이 가져왔는가 | ① Question-Retrieval 축 |
| **Answer Relevancy** | 답변이 질문에 실제로 부합하는가 | ② Question-Answer 축 |

RAGAS는 이 외에도 Context Entity Recall, Noise Sensitivity 등 추가 지표를 제공하지만, 위 4개가 가장 기본적으로 쓰이는 핵심 지표야.

---

## 채점 방식 — 결국 LLM-as-a-Judge

RAGAS의 이 지표들도 내부적으로는 [02번 파일](02_LLM_단일출력_평가자.md)에서 배운 **LLM-as-a-Judge** 원리로 동작해. 예를 들어 Faithfulness는:

1. 답변에서 주장(claim)들을 추출해달라고 LLM에게 요청
2. 각 주장이 컨텍스트에 실제로 근거가 있는지 LLM이 판단
3. 근거 있는 주장 비율을 점수로 환산

즉 **"RAG 전용 지표"라고 해서 완전히 새로운 채점 방법이 있는 게 아니라, 02번 파일에서 배운 LLM-as-a-Judge 방식을 RAG 상황에 맞춰 정교하게 구현한 것**이라고 이해하면 돼.

---

## RAGAS vs LangChain 범용 평가자 vs LangSmith — 최종 정리

| | RAGAS | LangChain 평가자 ([02번](02_LLM_단일출력_평가자.md)) | LangSmith ([04번](04_랭스미스_운영_평가기능.md)) |
|---|---|---|---|
| 특화 대상 | **RAG 전용** | 범용 LLM 앱 | 평가 운영 전반 |
| 지표 | Faithfulness, Context Precision/Recall, Answer Relevancy | QA, Criteria 등 | (지표 없음, 실행/기록/비교 인프라) |
| 조합 방식 | RAGAS 지표를 LangSmith `evaluate()`에 꽂아서 함께 쓰는 게 실무 표준 |||

---

## 왜 이 파트가 "래가스로 데이터셋 평가"(01번)로 시작해서 "래가스로 RAG 평가"(여기)로 끝났는가

1. **먼저 좋은 시험지가 있어야 한다** → 01번에서 합성 데이터셋을 만들고 품질을 검수
2. **채점하는 법(어휘)을 알아야 한다** → 02, 03번에서 LLM-as-a-Judge / 휴리스틱 등 채점 방법을 배움
3. **그 채점을 실제로 운영할 줄 알아야 한다** → 04번에서 LangSmith로 실험 비교/집계/반복/자동화를 배움
4. **이제 1~3을 다 합쳐서, 진짜 목적지인 "내 RAG 파이프라인 평가"를 완수한다** → 여기(05번)

RAGAS가 처음과 끝에 다시 등장한 건 우연이 아니라, **"시험지 준비" → "채점 인프라 학습" → "실제 채점"**이라는 자연스러운 학습 곡선의 시작점과 종착점에 같은 도구가 서 있었기 때문이야.

---

## 실무 사용 여부

| 항목 | 실무 사용 |
|---|---|
| RAGAS 4대 지표(Faithfulness, Context Precision/Recall, Answer Relevancy) | **RAG 평가의 사실상 표준**. 2026년 기준 비교 자료들에서도 "RAG 실패 유형에 가장 근접한 지표"로 꼽힘 |
| RAGAS 자체의 확장성 | RAG 영역 밖으로는 의도적으로 확장하지 않는 것으로 알려짐(집중형 라이브러리) — RAG 아닌 평가는 DeepEval, TruLens 등 다른 도구가 담당하는 경우가 많음 |
| 해커톤 활용 포인트 | "우리 RAG의 Faithfulness가 0.85"보다 "합성 데이터셋으로 설계 근거를 만들고, 반복평가로 안정성까지 확인했다"는 **전체 파이프라인 스토리**로 보여주면 설득력이 커짐 |
