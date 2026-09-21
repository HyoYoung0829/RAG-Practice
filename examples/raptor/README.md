# RAPTOR 예제 읽는 순서

먼저 `raptor.py`의 `main()`을 읽으세요. PDF 준비, 요약 트리 생성, 검색 인덱스 생성, 질문 답변의 네 단계로 실행됩니다. 파일 상단에서 PDF 경로, 청크 크기, 트리 레벨 수와 질문을 바꿀 수 있습니다.

| 모듈 | 담당하는 일 |
| --- | --- |
| `modules/documents.py` | PDF 읽기, 페이지 합치기, 작은 청크로 나누기 |
| `modules/models.py` | 임베딩 모델, 파일 캐시, LLM 만들기 |
| `modules/tree.py` | 텍스트 임베딩, 그룹별 요약, 상위 레벨 생성 |
| `modules/clustering.py` | UMAP 차원 축소와 GMM 그룹 배정 |
| `modules/retrieval.py` | FAISS 저장과 검색 기반 질문 답변 |
| `modules/tokens.py` | 토큰 수 계산 |

청크는 원문을 작게 나눈 조각입니다. 비슷한 청크끼리 묶어 요약하면 한 레벨이 만들어지고, 그 요약문들을 다시 묶어 요약하면 다음 레벨이 만들어집니다. 검색할 때는 원문 청크와 모든 레벨의 요약문을 함께 사용합니다.

`tree.py`의 함수들은 사용할 임베딩 모델과 LLM을 인자로 받습니다. 따라서 메인 파일에서 만든 모델이 어느 단계로 전달되는지 따라가며 읽을 수 있습니다. 클러스터링 세부 수식은 `clustering.py`에서 따로 살펴보세요.

저장소 루트에서 실행합니다. 기존 예제처럼 `.env`의 API 설정과 설치된 의존성이 필요합니다.

```bash
python3 examples/raptor/raptor.py
```
