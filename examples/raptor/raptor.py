"""실행 순서: PDF 준비 → 요약 트리 생성 → 검색 인덱스 생성 → 질문 답변."""

from dotenv import load_dotenv
from langchain_teddynote.messages import stream_response

from modules.documents import combine_pages, load_pdf_texts, split_into_chunks
from modules.models import create_embeddings, create_llm
from modules.retrieval import build_vectorstore, create_rag_chain
from modules.tokens import num_tokens_from_string
from modules.tree import collect_tree_texts, recursive_embed_cluster_summarize

PDF_PATH = "data/SPRI_AI_Brief_2023년12월호_F.pdf"
CHUNK_SIZE = 100
TREE_LEVELS = 3
CACHE_DIR = "./cache/"
INDEX_PATH = "RAPTOR"
QUESTIONS = [
    "전체 문서가 다루는 주요 내용에 대해 정리해주세요.",
    "Anthropic 에 투자 관련된 내용을 요약하세요.",
    "삼성전자가 개발한 생성형 AI 의 이름과 발표일은?",
]


def main():
    load_dotenv()

    # 1. PDF를 읽고 작은 청크(트리의 최하위 노드)로 나눕니다.
    # page_texts = 문서의 페이지 수.
    page_texts = load_pdf_texts(PDF_PATH)

    # full_text = 문서의 모든 텍스트 이어 붙인거.
    full_text = combine_pages(page_texts)

    # 리프 노드의 청크 사이즈.
    leaf_texts = split_into_chunks(full_text, chunk_size=CHUNK_SIZE)

    # 2. 비슷한 청크를 묶어 요약하고, 요약을 다시 묶어 트리를 만듭니다.

    embeddings = create_embeddings(CACHE_DIR)
    llm = create_llm()

    # 전체 재귀 실행. 부모 하나 나올때까지.
    results = recursive_embed_cluster_summarize(
        leaf_texts, embeddings, llm, n_levels=TREE_LEVELS
    )

    # 3. 원본 청크와 모든 요약문을 하나의 검색 인덱스에 넣습니다.
    # 생성된 청크 평탄화.
    all_texts = collect_tree_texts(leaf_texts, results)
    # 평탄화 한 청크 벡터스토어에 다 때려 넣기.
    vectorstore = build_vectorstore(all_texts, embeddings, INDEX_PATH)
    # 체인 생성.
    rag_chain = create_rag_chain(vectorstore, llm)

    # 4. 관련된 원문/요약문을 검색해 질문에 답변합니다.
    for question in QUESTIONS:
        stream_response(rag_chain.stream(question))


if __name__ == "__main__":
    main()
