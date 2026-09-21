"""청크를 묶어 요약하고 그 요약을 다시 묶어 상위 레벨을 만듭니다."""

# 각 레벨의 처리 순서:
# 입력 텍스트 -> 임베딩 -> 글로벌 UMAP/GMM -> 글로벌 그룹별 로컬 UMAP/GMM
# -> 최종 로컬 그룹별 LLM 요약 -> 그 요약문을 다음 레벨의 입력으로 전달
# 레벨 1의 입력은 원본 청크, 레벨 2부터는 바로 이전 레벨의 요약문입니다.
# 글로벌/로컬은 한 레벨 안의 그룹 분할 단계이며 각각 별도의 트리 레벨이 아닙니다.
# 실제 글로벌/로컬 처리는 clustering.py의 perform_clustering() 안에서 실행됩니다.

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .clustering import perform_clustering


# 텍스트 목록을 임베딩 모델로 숫자 벡터로 바꾸고 NumPy 배열로 반환합니다.
def embed(texts, embeddings):
    text_embeddings = embeddings.embed_documents(texts)

    # 클러스터링 함수들(UMAP, GMM)이 numpy 배열을 기대하므로 형변환
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


# 현재 레벨의 텍스트를 임베딩한 뒤 글로벌 -> 로컬 클러스터링을 모두 실행합니다.
# 반환 표의 cluster는 글로벌 그룹 번호가 아니라 최종 로컬 그룹의 고유 번호입니다.
def embed_cluster_texts(texts, embeddings):
    # 임베딩 생성
    text_embeddings_np = embed(texts, embeddings)
    # perform_clustering() 내부 순서 (이 함수 호출 하나에 두 단계가 들어 있습니다):
    # 1. 글로벌: 현재 레벨의 전체 벡터 -> UMAP 차원 축소 -> GMM으로 큰 그룹 배정.
    # 2. 로컬: 각 글로벌 그룹의 원래 벡터 -> UMAP 차원 축소 -> GMM으로 세부 그룹 배정.
    # 3. 로컬 그룹 번호가 글로벌 그룹 간 겹치지 않게 조정해 반환.
    # 여기서는 dim=10, threshold=0.1입니다. 그룹 소속 확률이 0.1을 넘으면 배정합니다.
    # 전체 벡터가 11개 이하면 두 단계 모두 생략하고 하나의 그룹으로 처리합니다.
    # 개별 글로벌 그룹이 11개 이하면 그 그룹의 로컬 분할만 생략합니다.
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
    # 결과를 저장할 DataFrame 초기화
    df = pd.DataFrame()
    # 원본 텍스트 저장
    df["text"] = texts
    # DataFrame에 리스트로 임베딩 저장
    df["embd"] = list(text_embeddings_np)
    # 클러스터 라벨 저장 (한 텍스트가 여러 클러스터에 속할 수 있어 배열 형태)
    df["cluster"] = cluster_labels
    return df


# 같은 그룹에 속한 텍스트들을 구분자로 이어 요약할 문자열을 만듭니다.
def fmt_txt(df: pd.DataFrame) -> str:
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


# 한 레벨을 만듭니다: 임베딩 -> 글로벌 분할 -> 로컬 분할 -> 최종 로컬 그룹별 요약.
# level=1이면 원본 청크를, level>=2이면 이전 레벨의 요약문을 처리합니다.
# 반환값은 입력 텍스트의 그룹 배정 표(df_clusters)와 이번 레벨의 요약 표(df_summary)입니다.
def embed_cluster_summarize_texts(
    texts: List[str], level: int, embeddings, llm
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 1. 임베딩과 글로벌/로컬 분할을 완료해 최종 로컬 그룹 배정 표를 받습니다.
    df_clusters = embed_cluster_texts(texts, embeddings)

    # 2. 한 텍스트가 여러 최종 로컬 그룹에 속하면 그룹마다 행을 하나씩 만듭니다.
    # 예: text=A, cluster=[0, 2] -> (A, 0), (A, 2). 두 그룹 모두 A를 요약에 사용합니다.
    expanded_list = []

    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # 확장된 목록에서 새 데이터프레임을 생성합니다.
    expanded_df = pd.DataFrame(expanded_list)

    # 요약 대상은 글로벌 그룹이 아니라 글로벌 -> 로컬 분할을 마친 최종 그룹입니다.
    all_clusters = expanded_df["cluster"].unique()

    # [주의] 아래 프롬프트는 "LangChain 표현 언어(LCEL) 문서를 요약하라"는 내용으로 고정되어 있습니다.
    # 이는 이 튜토리얼이 원래 다른 예제(LCEL 문서)에서 가져온 템플릿을 그대로 쓴 흔적으로 보이며,
    # 실제로는 지금 다루는 PDF(SPRI AI 브리프)와 무관한 문구입니다. 그대로 둬도 LLM이 맥락({context})을
    # 보고 어느 정도 알아서 요약하긴 하지만, 실전(특히 해커톤에서 다른 도메인 데이터를 쓸 때)에는
    # 이 템플릿을 실제 문서 주제에 맞게 고쳐 쓰는 것을 권장합니다.
    template = """여기 LangChain 표현 언어 문서의 하위 집합이 있습니다.
    
    LangChain 표현 언어는 LangChain에서 체인을 구성하는 방법을 제공합니다.
    
    제공된 문서의 자세한 요약을 제공하십시오.
    
    문서:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 3. 최종 로컬 그룹마다 텍스트를 합쳐 LLM으로 요약문 하나를 만듭니다.
    # 글로벌과 로컬 분할을 모두 끝낸 뒤 요약합니다. 두 분할 사이에는 요약하지 않습니다.
    # 요약문은 현재 level의 노드로 저장되고 다음 level의 입력이 됩니다.
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # 요약, 해당 클러스터 및 레벨을 저장할 데이터프레임을 생성합니다.
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


# 각 레벨에서 [임베딩 -> 글로벌 -> 로컬 -> 요약] 전체 과정을 실행하는 재귀 함수입니다.
# 레벨 1: 원본 청크 -> 글로벌/로컬 분할 -> 레벨 1 요약문.
# 레벨 2: 레벨 1 요약문 -> 새 임베딩과 글로벌/로컬 분할 -> 레벨 2 요약문.
# 레벨 3: 레벨 2 요약문으로 같은 과정을 반복합니다.
# 최대 레벨에 도달하거나 현재 레벨의 요약 그룹이 1개 이하이면 종료합니다.
def recursive_embed_cluster_summarize(
    texts: List[str], embeddings, llm, level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    # 각 레벨에서의 결과를 저장할 사전
    results = {}

    # 현재 레벨에서 임베딩 -> 글로벌 분할 -> 로컬 분할 -> 요약을 모두 수행합니다.
    df_clusters, df_summary = embed_cluster_summarize_texts(
        texts, level, embeddings, llm
    )

    # 현재 레벨의 결과 저장
    results[level] = (df_clusters, df_summary)

    # 추가 재귀가 가능하고 의미가 있는지 결정
    unique_clusters = df_summary["cluster"].nunique()

    # 현재 레벨이 최대 레벨보다 낮고, 유니크한 클러스터가 1개 초과인 경우에만 계속 재귀
    if level < n_levels and unique_clusters > 1:
        # 이번 레벨의 요약문을 다음 레벨의 입력으로 넘깁니다.
        # 재귀 호출에서 새 임베딩과 글로벌/로컬 분할을 다시 수행합니다.
        # 이전 레벨의 그룹 번호나 축소 벡터를 재사용하지 않습니다.
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, embeddings, llm, level + 1, n_levels
        )

        # 다음 레벨의 결과를 현재 결과 사전에 병합
        results.update(next_level_results)

    return results


# 검색 대상으로 사용할 원본 청크와 모든 레벨의 요약문을 하나의 목록으로 모읍니다.
def collect_tree_texts(leaf_texts, results):
    all_texts = leaf_texts.copy()
    for level in sorted(results):
        all_texts.extend(results[level][1]["summaries"].tolist())
    return all_texts
