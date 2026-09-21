"""청킹 튜닝 실험. chunk_size만 800(rag_pipeline.py) -> 300으로 줄이고 나머지는 동일.

질문 0("AI 로보틱스 시장 2030년 전망") 실패 원인이 800자 청크 안에 서로 다른
3개 소주제(트렌드 서두/시장 통계/정책 투자)가 섞여 임베딩이 희석된 것으로 확인됐음
(docs/07 참고). 청크를 불릿 하나 정도 크기로 줄여서 주제가 안 섞이게 하면 개선되는지
확인하는 게 목적.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import build_retriever, build_vectorstore, load_documents  # noqa: F401 (재사용, 변경 없음)

# 페이지 인용("(출처: N페이지)") 요구를 뺀 버전. LLM에게 준 문맥과 RAGAS Faithfulness에
# 넘기는 contexts가 어긋나던 문제(질문 4가 항상 0.5로 나오던 원인 — RAGAS는 페이지 번호
# 없는 순수 본문만 받는데 답변은 "2페이지"를 인용해서, 그 인용 자체가 검증 불가능한
# 주장으로 잡혀 감점됨)를 프롬프트에서 인용 요구를 없애 해결.
PROMPT_PATH = Path(__file__).parent / "prompt" / "rag-answer-tuned.yaml"


def split_documents(docs: list[Document]) -> list[Document]:
    """2단계(튜닝): chunk_size=300, chunk_overlap=100 (실험 4: 50에서 확대).
    Context Precision/Recall이 이미 0.97~1.00으로 천장이라 큰 변화는 기대하기 어렵지만,
    오버랩 확대가 실제로 영향을 주는지(개선/무변화/악화) 확인하려는 실험."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return splitter.split_documents(docs)


def build_pipeline(vectorstore: FAISS, use_reranker: bool = False, k: int = 4):
    """4~5단계: rag_pipeline.py와 동일한 리트리버 로직 재사용, 프롬프트만 교체."""
    retriever = build_retriever(vectorstore, use_reranker=use_reranker, k=k)
    prompt = load_prompt(str(PROMPT_PATH), encoding="utf-8")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    def ask(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "contexts": [doc.page_content for doc in docs]}

    return ask
