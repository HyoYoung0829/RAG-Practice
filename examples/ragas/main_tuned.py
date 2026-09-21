from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from rag_pipeline_tuned import build_pipeline, build_vectorstore, load_documents, split_documents

load_dotenv()

docs = load_documents()
chunks = split_documents(docs)  # chunk_size=300 (rag_pipeline_tuned.py) — 유일하게 바뀐 변수
vectorstore = build_vectorstore(chunks)

# main.py와 완전히 동일한 평가셋/채점 로직. 바뀐 건 chunk_size 하나뿐이라는 걸
# 보여주려고 일부러 그대로 복붙함 (docs/07에서 리랭커 유무만 바꿨던 것과 같은 원칙).
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

# RAGAS 채점 LLM을 temperature=0으로 고정 (docs/07 "채점 노이즈" 참고).
# 완전한 결정론은 아니지만, 앞으로의 튜닝 실험에서 "진짜 효과 vs 우연"을 덜 헷갈리게 함.
JUDGE_LLM = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


def run_eval(name: str, ask) -> None:
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

    scores = evaluate(EvaluationDataset(samples=samples), metrics=METRICS, llm=JUDGE_LLM)
    result_df = scores.to_pandas()

    print(f"\n=== {name} ===")
    print(result_df[["user_input"] + [m.name for m in METRICS]].to_string(index=False))
    print("--- 평균 ---")
    print(result_df[[m.name for m in METRICS]].mean())


print(f"청크 개수: {len(chunks)}개 (chunk_size=300 기준, 기존 106개와 비교)")

ask_baseline = build_pipeline(vectorstore, use_reranker=False)
run_eval("청킹 튜닝 + 베이스라인 (리랭커 없음)", ask_baseline)

ask_reranked = build_pipeline(vectorstore, use_reranker=True)
run_eval("청킹 튜닝 + 리랭커 적용", ask_reranked)
