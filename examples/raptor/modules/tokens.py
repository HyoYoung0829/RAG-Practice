# 텍스트를 모델이 처리하는 단위인 토큰으로 나누어 개수를 셉니다.
# raptor.py에서 문서 전체의 토큰 수를 출력할 때 사용합니다.

import tiktoken


# 지정한 인코딩으로 텍스트를 나누어 토큰 개수를 셉니다.
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """텍스트를 지정한 인코딩 기준의 토큰 수로 계산합니다."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))
