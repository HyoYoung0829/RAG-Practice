"""UMAP으로 벡터 차원을 줄이고 GMM으로 비슷한 주제를 묶습니다."""
# 텍스트의 의미를 나타내는 벡터를 받아 비슷한 것끼리 그룹으로 나눕니다.
# UMAP으로 비교할 숫자의 개수를 줄이고, GMM으로 각 벡터가 속할 그룹을 정합니다.

from typing import List, Optional

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 42  # 재현성을 위한 고정된 시드 값


# 전체 벡터의 차원을 줄여 큰 주제별 그룹을 찾기 쉽게 만듭니다.
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """[Global 차원 축소]
    전체 임베딩(예: leaf 청크 전부, 혹은 특정 레벨의 요약문 전부)을 대상으로
    UMAP을 적용해 고차원 벡터를 저차원(dim)으로 압축합니다.

    의도: 임베딩은 보통 수백~수천 차원이라 "차원의 저주" 때문에 클러스터링이 잘 안 먹힙니다.
    클러스터링 전에 먼저 차원을 확 줄여서 GMM이 그룹 구조를 더 잘 찾게 도와주는 전처리 단계입니다.
    "Global"이라는 이름은 이 축소가 "전체 데이터 집합"을 기준으로 이루어진다는 뜻입니다
    (이후 local_cluster_embeddings는 이미 한 번 나뉜 하위 그룹 안에서만 다시 이 작업을 반복합니다).

    Args:
        embeddings (np.ndarray): 차원을 축소할 임베딩 벡터들
        dim (int): 축소할 차원의 수
        n_neighbors (Optional[int], optional): UMAP에서 사용할 이웃의 수. 기본값은 None으로, 이 경우 데이터 크기에 따라 자동 계산됨
        metric (str, optional): 거리 계산에 사용할 메트릭. 기본값은 "cosine"

    Returns:
        np.ndarray: 차원이 축소된 임베딩 벡터들
    """
    # 이웃 수 계산 (데이터 개수의 제곱근 정도로 자동 설정 — UMAP의 일반적인 경험적 기준)
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    # UMAP 적용
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


# 큰 그룹 하나에 속한 벡터의 차원을 줄여 세부 그룹을 찾기 쉽게 만듭니다.
def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """[Local 차원 축소]
    global_cluster_embeddings로 이미 한 번 큰 그룹(global cluster)으로 나뉜 뒤,
    그 그룹 하나에 속한 데이터만 떼어내서 다시 UMAP을 적용하는 함수입니다.

    의도: "법률 vs 기술"처럼 스케일이 큰 차이와, "LLM vs 이미지생성"처럼 스케일이 작은 차이를
    하나의 클러스터링으로 동시에 잘 구분하기는 어렵습니다. 그래서 먼저 큰 틀(global)로 나누고,
    그 안에서 다시 세밀하게(local) 나누는 2단계 구조를 씁니다.
    num_neighbors 기본값(10)이 global의 자동 계산 방식과 다른 이유는,
    이미 데이터가 작은 하위 그룹으로 좁혀진 상태라 이웃 수를 고정된 작은 값으로 둬도 충분하기 때문입니다.

    [수정 안내] 원본 노트북에는 이 함수 본문이 `return`만 있고 실제 UMAP 호출이 비어 있어
    항상 None을 반환하는 상태였습니다(미완성 코드로 추정). global_cluster_embeddings와
    동일한 로직에 num_neighbors만 다르게 적용하도록 아래처럼 채워 넣었습니다.

    Args:
        embeddings (np.ndarray): 차원을 축소할 임베딩 벡터들
        dim (int): 축소할 차원의 수
        num_neighbors (int, optional): UMAP에서 사용할 이웃의 수. 기본값은 10
        metric (str, optional): 거리 계산에 사용할 메트릭. 기본값은 "cosine"

    Returns:
        np.ndarray: 차원이 축소된 임베딩 벡터들
    """
    # UMAP 적용 (global과 동일한 방식이되, 이웃 수는 고정된 num_neighbors 사용)
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


# 그룹 개수를 바꿔가며 BIC 점수를 비교하고 가장 낮은 점수의 개수를 선택합니다.
def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """BIC(베이지안 정보 기준) 점수를 기반으로 최적의 클러스터 수를 찾는 함수입니다.

    원리: 클러스터 개수를 1개부터 max_clusters까지 하나씩 바꿔가며 GMM을 학습시키고,
    각 경우의 BIC 점수를 비교합니다. BIC가 낮을수록 "적은 파라미터로 데이터를 잘 설명한다"는
    뜻이라, 가장 낮은 BIC를 주는 클러스터 개수를 "최적"으로 선택합니다.
    → 이 덕분에 "클러스터를 몇 개로 나눌지"를 사람이 미리 정하지 않아도 됩니다.

    참고: BIC 점수 자체(gm.bic())는 scikit-learn이 계산해주는 내장 기능이고,
    "여러 후보를 반복 비교해서 최적값을 고르는" 이 탐색 로직은 RAPTOR 저자가 직접 구현한 부분입니다.

    Args:
        embeddings (np.ndarray): 클러스터링할 임베딩 벡터들
        max_clusters (int, optional): 탐색할 최대 클러스터 수. 기본값은 50
        random_state (int, optional): 난수 생성을 위한 시드값. 기본값은 RANDOM_SEED

    Returns:
        int: BIC 점수가 가장 낮은(최적의) 클러스터 수
    """
    # 최대 클러스터 수와 임베딩의 길이 중 작은 값을 최대 클러스터 수로 설정
    # (데이터 개수보다 클러스터 개수가 더 많을 수는 없으므로)
    max_clusters = min(max_clusters, len(embeddings))
    # 1부터 최대 클러스터 수까지의 범위를 생성
    n_clusters = np.arange(1, max_clusters)

    # BIC 점수를 저장할 리스트
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        # 학습된 모델의 BIC 점수를 리스트에 추가
        bics.append(gm.bic(embeddings))

    # BIC 점수가 가장 낮은 클러스터 수를 반환
    return n_clusters[np.argmin(bics)]


# 각 벡터가 그룹에 속할 확률을 계산하고 기준 확률을 넘는 그룹들을 배정합니다.
def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """GMM(가우시안 혼합 모델)을 이용해 주어진 임베딩에 대해 클러스터를 할당합니다.

    원리: GMM은 K-means처럼 "무조건 하나의 클러스터에 배정"하는 게 아니라,
    "이 데이터가 클러스터 A에 속할 확률 70%, B에 속할 확률 30%" 처럼 확률(soft clustering)로
    결과를 냅니다. 그래서 threshold를 넘는 확률을 가진 클러스터에는 전부 배정할 수 있고,
    결과적으로 하나의 텍스트 조각이 여러 클러스터에 동시에 속하는 것도 허용됩니다.
    (법률 조항 하나가 "계약"과 "손해배상" 두 주제에 걸쳐 있을 수 있는 것처럼, 텍스트는
    원래 하나의 카테고리로만 깔끔히 나뉘지 않는 경우가 많기 때문에 이 유연함이 유용합니다.)

    Args:
        embeddings: 클러스터링할 임베딩 벡터들
        threshold: 이 확률을 넘는 클러스터에만 소속시킴
        random_state: 재현성을 위한 시드값

    Returns:
        labels: 각 데이터 포인트가 속한 클러스터 인덱스 배열들의 리스트
        n_clusters: 사용된 클러스터 개수
    """
    # 최적의 클러스터 수 산정 (위에서 정의한 BIC 기반 함수 사용)
    n_clusters = get_optimal_clusters(embeddings)

    # 가우시안 혼합 모델을 초기화
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)

    # 임베딩이 각 클러스터에 속할 확률을 예측
    probs = gm.predict_proba(embeddings)

    # 임계값을 초과하는 확률을 가진 클러스터를 레이블로 선택 (여러 개일 수 있음)
    labels = [np.where(prob > threshold)[0] for prob in probs]

    # 레이블과 클러스터 수를 반환
    return labels, n_clusters


# 전체 벡터를 큰 그룹으로 나눈 뒤 각 그룹을 세분화해 최종 그룹 번호를 반환합니다.
def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """전역(Global) → 지역(Local) 2단계로 클러스터링을 수행하는 핵심 파이프라인 함수입니다.

    전체 흐름 (책 비유: 먼저 "챕터"로 나누고, 각 챕터 안에서 다시 "절"로 나누는 것과 동일):
      1. 전체 임베딩을 UMAP으로 축소 (global_cluster_embeddings)
      2. 축소된 벡터로 GMM 클러스터링 → 큰 그룹(global cluster) 여러 개 확보
      3. 각 global cluster 하나하나에 대해:
         3-1. 그 그룹에 속한 임베딩만 추려서 다시 UMAP 축소 (local_cluster_embeddings)
         3-2. 다시 GMM 클러스터링 → 그 그룹 안의 세부 그룹(local cluster) 확보
      4. 모든 local cluster에 전역적으로 겹치지 않는 고유 ID를 부여해서 최종 반환

    이렇게 global/local 2단계로 나누는 이유는 GMM이라는 알고리즘 자체의 특성이라기보다,
    "큰 스케일 차이"와 "작은 스케일 차이"를 한 번의 클러스터링으로 동시에 잘 구분하기 어렵기
    때문에 쓰는 범용적인 다중 스케일 클러스터링 기법입니다.

    Args:
        embeddings: 클러스터링할 임베딩 벡터들
        dim: 차원 축소 시 목표 차원 수
        threshold: GMM 클러스터링에서 사용할 확률 임계값

    Returns:
        List[np.ndarray]: 각 데이터 포인트가 속한 (전역+지역 통합) 클러스터 ID 리스트
    """

    if len(embeddings) <= dim + 1:
        # 데이터가 너무 적으면(축소할 차원보다도 적으면) UMAP이 의미가 없으므로
        # 클러스터링을 생략하고 전부 같은 클러스터(0번)로 처리합니다.
        return [np.array([0]) for _ in range(len(embeddings))]

    # 1~2단계: 글로벌 차원 축소 + 글로벌 클러스터링 (큰 카테고리로 나누기)
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    # 로컬 클러스터링을 위한 초기화
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # 3단계: 각 글로벌 클러스터를 순회하며 그 안에서 로컬 클러스터링 수행
    for i in range(n_global_clusters):
        # 현재 글로벌 클러스터에 속하는 임베딩만 추출
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # 이 글로벌 클러스터 자체가 너무 작으면 더 세분화하지 않고 하나로 처리
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 로컬 차원 축소 및 로컬 클러스터링 (그룹 내부를 세부적으로 나누기)
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # 로컬 클러스터 ID가 전체적으로 겹치지 않도록 total_clusters만큼 오프셋을 더해 부여
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters
