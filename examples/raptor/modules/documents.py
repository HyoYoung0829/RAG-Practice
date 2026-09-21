"""PDF를 읽고 트리의 최하위 노드가 될 텍스트 청크를 준비합니다."""
# PDF 페이지의 텍스트를 합친 뒤, 토큰 수 기준으로 작은 조각(청크)으로 나눕니다.
# 여기서 만든 청크가 tree.py의 입력이 됩니다.

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PDF를 읽고 각 페이지의 텍스트를 목록으로 반환합니다.
def load_pdf_texts(pdf_path):
    """페이지별 텍스트를 반환합니다. 기존 예제의 정렬 순서를 유지합니다."""
    docs = PDFPlumberLoader(pdf_path).load()
    docs = list(reversed(sorted(docs, key=lambda doc: doc.metadata["source"])))
    return [doc.page_content for doc in docs]


# 페이지별 텍스트를 구분자로 이어 하나의 긴 문자열로 만듭니다.
def combine_pages(page_texts):
    """페이지들을 구분자로 이어 하나의 텍스트로 만듭니다."""
    return "\n\n\n --- \n\n\n".join(page_texts)


# 긴 텍스트를 지정한 토큰 수 기준으로 작은 조각(청크)으로 나눕니다.
def split_into_chunks(text, chunk_size=100):
    """텍스트를 토큰 수 기준으로 작은 청크로 나눕니다."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    return splitter.split_text(text)
