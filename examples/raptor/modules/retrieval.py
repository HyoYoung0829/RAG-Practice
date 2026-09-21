"""원문과 요약문을 검색 인덱스에 저장하고 질문 답변 체인을 만듭니다."""
# 원본 청크와 요약문을 FAISS(벡터 검색 저장소)에 넣고 로컬 파일로 저장합니다.
# 질문과 관련된 문서를 검색한 뒤, 그 내용을 LLM에 전달해 답변을 생성합니다.

import os

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# 원문과 요약문의 검색 인덱스를 만들고 기존 인덱스가 있으면 병합해 저장합니다.
def build_vectorstore(texts, embeddings, index_path="RAPTOR"):
    """새 인덱스를 만들고 기존 인덱스가 있으면 병합해 저장합니다."""
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    if os.path.exists(index_path):
        # 직접 생성한 신뢰할 수 있는 로컬 인덱스만 사용합니다.
        local_index = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        local_index.merge_from(vectorstore)
        local_index.save_local(index_path)
    else:
        vectorstore.save_local(folder_path=index_path)
    return vectorstore


# 검색된 문서의 내용을 문서 태그로 감싸 하나의 문자열로 합칩니다.
def format_docs(docs):
    """검색한 문서들을 LLM에 전달할 텍스트로 합칩니다."""
    return "\n\n".join(f"<document>{doc.page_content}</document>" for doc in docs)


# 질문으로 문서를 검색하고 검색 결과를 LLM에 전달해 답변하는 체인을 만듭니다.
def create_rag_chain(vectorstore, llm):
    """질문 → 검색 → 프롬프트 → LLM 답변 순서로 연결합니다."""
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate.from_template("""
        You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
    You are given PDF documents. Your primary mission is to answer questions based on provided context.
    Ensure your response is concise and directly addresses the question without any additional narration.
    
    ###
    
    Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names).
    
    # Steps
    
    1. Carefully read and understand the context provided.
    2. Identify the key information related to the question within the context.
    3. Formulate a concise answer based on the relevant information.
    4. Ensure your final answer directly addresses the question.
    
    # Output Format:
    [General introduction of the answer]
    [Comprehensive answer to the question]
    
    ###
    
    Remember:
    - It's crucial to base your answer solely on the **PROVIDED CONTEXT**. 
    - DO NOT use any external knowledge or information not present in the given materials.
    
    ###
    
    # Here is the user's QUESTION that you should answer:
    {question}
    
    # Here is the CONTEXT that you should use to answer the question:
    {context}
    
    [Note]
    - Answer should be written in Korean.
    
    # Your final ANSWER to the user's QUESTION:""")
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
