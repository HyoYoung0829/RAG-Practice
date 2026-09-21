# RAG Practice Lab

RAG와 LangChain 관련 실습을 폴더별로 쌓아가기 위한 저장소입니다.

## 폴더 구조

```text
.
├── examples/
│   ├── 01-basic-chatbot/
│   │   ├── app.py
│   │   └── README.md
│   └── 02-rag-placeholder/
│       └── README.md
├── shared/
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   └── README.md
├── pyproject.toml
├── uv.lock
└── .env
```

## 실행 방법

먼저 루트에서 의존성을 설치합니다.

```bash
uv sync
```

실습 앱은 각 실습 폴더의 `app.py`를 직접 지정해서 실행합니다.

```bash
uv run streamlit run examples/01-basic-chatbot/app.py
```

새 패키지는 루트에서 추가합니다.

```bash
uv add 패키지명
```

## 실습 추가 규칙

새 실습은 `examples/번호-이름/` 형태로 추가합니다.

```text
examples/
└── 03-vector-search/
    ├── app.py
    ├── README.md
    └── assets/
```

공통으로 재사용하는 함수나 설정은 `shared/`에 두고, 여러 실습에서 함께 쓰는 샘플 문서는 `data/`에 둡니다.
