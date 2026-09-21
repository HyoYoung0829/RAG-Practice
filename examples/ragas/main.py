from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from rag_pipeline import build_pipeline, build_vectorstore, load_documents, split_documents

load_dotenv()

docs = load_documents()
chunks = split_documents(docs)
vectorstore = build_vectorstore(chunks)
ask = build_pipeline(vectorstore)

# 6. RAGAS 평가 준비
# data/피지컬 AI의 현황과 시사점.pdf 를 직접 읽고 작성한 평가 질문 5개 + 정답(reference).
# TestsetGenerator(자동 생성)를 시도했으나 이 문서 규모에서는 시나리오가 0개로 조용히
# 실패함(멀티홉 질문에 필요한 노드 간 관계가 기본 임계값을 못 넘긴 것으로 추정). 우리
# 목적은 "완벽한 데이터셋"이 아니라 "리랭커 有/無를 같은 잣대로 비교"라 수기 평가셋으로 회귀.
EVAL_SET = [
    {
        "question": "AI 로보틱스 시장 규모는 2030년에 얼마로 전망되나요?",
        "ground_truth": "Statista에 따르면 2030년 AI 로보틱스 시장은 약 643억 달러(한화 약 85조 원)에 이를 것으로 전망된다.",
    },
    {
        "question": "피지컬 AI는 어떤 기술 요소들의 융합으로 정의되나요?",
        "ground_truth": "AI 기반 모델(두뇌), 컴퓨터 비전·센서(감각), 엣지 컴퓨팅 및 네트워크 인프라(연결), 제어 및 액추에이터(행동)의 융합으로 정의된다.",
    },
    {
        "question": "피지컬 AI는 기술 수준과 형태에 따라 어떻게 분류되나요?",
        "ground_truth": "휴머노이드형, 자율주행차형, 드론형, AGV & AMR형으로 분류된다.",
    },
    {
        "question": "피지컬 AI 확산을 가로막는 장애 요인은 무엇인가요?",
        "ground_truth": "막대한 연산 자원과 개발 비용, 물리 환경 적용의 기술적 제약, 노동시장 구조 변화, 법적 책임과 윤리 기준의 미비 등이 있다.",
    },
    {
        "question": "2020년 대비 2025년 AI 로보틱스 시장은 얼마나 성장했나요?",
        "ground_truth": "2020년 약 50억 달러에서 2025년 225억 달러로 350% 성장했다.",
    },
]

# ②단계: 내 RAG(베이스라인)를 5개 질문에 실제로 돌려서 "학생 답안"을 만든다.
# 아직 RAGAS로 채점(③단계)은 안 하고, 채점관에게 넘길 데이터 모양만 먼저 확인한다.
samples = []
for item in EVAL_SET:
    result = ask(item["question"])
    samples.append(
        SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=item["ground_truth"],
        )
    )

print(f"총 {len(samples)}개 샘플 생성됨\n")
for i, sample in enumerate(samples):
    print(f"===== [{i}] {sample.user_input} =====")
    print(f"답변  : {sample.response}")
    print(f"정답  : {sample.reference}")
    print(f"검색된 문맥 {len(sample.retrieved_contexts)}개:")
    for j, ctx in enumerate(sample.retrieved_contexts):
        preview = ctx.replace("\n", " ")[:80]
        print(f"  [{j}] {preview}...")
    print()

# 7. RAGAS로 실제 채점 (③단계: 채점관이 학생 답안 vs 정답을 비교)
METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
scores = evaluate(EvaluationDataset(samples=samples), metrics=METRICS)

result_df = scores.to_pandas()
print("=== 베이스라인 (리랭커 없음) 질문별 점수 ===")
print(result_df[["user_input"] + [m.name for m in METRICS]].to_string(index=False))
print("\n=== 평균 ===")
print(result_df[[m.name for m in METRICS]].mean())
