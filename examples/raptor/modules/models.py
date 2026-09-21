"""임베딩 모델과 요약/답변용 LLM을 생성합니다."""
# 임베딩 모델은 텍스트를 의미 비교용 숫자 벡터로 바꾸고, LLM은 요약과 답변을 만듭니다.
# 임베딩 결과를 파일에 저장해 같은 텍스트를 처리할 때 재사용합니다.

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 텍스트를 숫자 벡터로 바꾸는 모델을 만들고, 결과를 재사용할 파일 캐시를 붙입니다.
def create_embeddings(cache_dir="./cache/"):
    """같은 텍스트의 임베딩을 재사용하도록 파일 캐시를 붙입니다."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings, LocalFileStore(cache_dir), namespace=embeddings.model
    )


# 그룹별 요약과 질문 답변에 사용할 LLM을 만듭니다.
def create_llm():
    """요약과 답변에 사용할 모델을 만듭니다."""
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)
