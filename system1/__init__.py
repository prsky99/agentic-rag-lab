"""1장의 결정 계층(System 1) 단독 실행 패키지.

7장 이후의 하네스는 ``domains/rag/harness``에 있고, 이 패키지는 1장을 따라
게이트 4종만 먼저 실행해 보는 용도입니다. ``python3 -m system1.demo``로 실행해 보세요.

``client``는 httpx가 필요하므로 처음 쓸 때 읽습니다. 그래서 ``pip install pyyaml pytest``만
설치한 상태에서도 로컬 모드 데모(``python3 -m system1.demo``)가 실행됩니다.
"""

__all__ = ["DecisionClient", "DecisionUnavailable"]


def __getattr__(name: str):
    if name in __all__:
        from . import client
        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
