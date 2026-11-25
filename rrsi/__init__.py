"""RRSI(Regularized Recursive Self-Improvement of Agent Harnesses) 코어의 교육용 구현.

google-research/rrsi의 구조(라운드·후보·critic·노이즈 밴드·편집 예산)를 따라 이 책을 위해
새로 작성한 축소판입니다. 업스트림 코드를 그대로 옮긴 것이 아니며, 차이는 책 7~8장에 적었습니다.
도메인 작업에서는 이 디렉터리를 고치지 않습니다.
도메인은 ``rrsi.domain.Domain`` 구현만 제공하고, 편집은 각 도메인의
``harness/`` 디렉터리 안에서만 일어납니다(7~8장).
"""

__version__ = "0.1.0"
