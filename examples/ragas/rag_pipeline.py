"""PDF 로드 -> 청킹 -> 임베딩 -> 검색 -> 답변까지, 나이브 RAG 파이프라인 (1~5단계)."""

from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).parent / "data" / "피지컬 AI의 현황과 시사점.pdf"
PROMPT_PATH = Path(__file__).parent / "prompt" / "rag-answer.yaml"


def load_documents() -> list[Document]:
    """1단계: PDF 로드 (페이지 단위 Document 리스트)."""
    return PDFPlumberLoader(str(PDF_PATH)).load()


def split_documents(docs: list[Document]) -> list[Document]:
    """2단계: 청킹. 페이지(Document) 단위로 각각 쪼갠다.
    chunk_overlap이 페이지 경계는 못 넘지만, 대신 청크마다 원본 page 번호가 남아
    출처 추적(프롬프트의 "(출처: N페이지)")이 가능해짐."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """3단계: 임베딩 + 벡터스토어."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def build_pipeline(vectorstore: FAISS, k: int = 4):
    """4~5단계: 리트리버 + 프롬프트/LLM. question(str)을 받아 답변을 내는 함수를 돌려준다."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = load_prompt(str(PROMPT_PATH), encoding="utf-8")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    def ask(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[페이지 {doc.metadata['page']}]\n{doc.page_content}" for doc in docs
        )
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "contexts": [doc.page_content for doc in docs]}

    return ask
