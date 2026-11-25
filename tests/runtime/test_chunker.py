"""구조 청킹 회귀 시험(RT08): 코드 펜스는 여는 줄부터 닫는 줄까지 한 청크로 유지합니다."""
from __future__ import annotations

from services.ingest.chunker import split_structured

DOC = """# 설치

설명 문단입니다. 아래 명령을 실행합니다.

```bash
pip install -r requirements.txt

python -m system1.demo
```

마무리 문단입니다.
"""


def test_code_fence_is_one_chunk():
    chunks = split_structured(DOC)
    code = [c for c in chunks if "pip install" in c.text]
    assert len(code) == 1
    text = code[0].text
    assert "python -m system1.demo" in text and text.count("```") == 2   # 여는·닫는 펜스가 같은 청크
    assert all("마무리 문단" not in c.text for c in code)                 # 코드 뒤 산문은 따로
