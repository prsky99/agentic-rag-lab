"""게이트웨이의 신원 확인 (11.2절).

JWT의 tenant claim이 유일한 테넌트 출처입니다. 헤더나 본문의 tenant_id를 믿지
않습니다. claim이 없으면 403으로 거절합니다(fail-closed).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException

ISSUER = os.environ.get("JWT_ISSUER", "rag-dev")
AUDIENCE = os.environ.get("JWT_AUDIENCE", "rag-api")
# 운영에서는 ExternalSecret으로 주입합니다. 아래 기본값은 로컬 kind 전용입니다.
SECRET = os.environ.get("JWT_SECRET", "dev-only-secret")
JWKS_URL = os.environ.get("JWKS_URL", "")


@dataclass(frozen=True)
class Tenant:
    id: str
    tier: str
    user: str


@lru_cache(maxsize=4)
def _jwks_client(url: str):
    """JWKS 클라이언트는 URL마다 하나만 만들어 재사용합니다(공개키 캐시 유지)."""
    from jwt import PyJWKClient
    return PyJWKClient(url, cache_keys=True)


def _required_claims() -> list[str]:
    """만료(exp)·주체(sub)는 항상, 발급자·대상은 설정돼 있을 때 필수입니다."""
    req = ["exp", "sub"]
    if ISSUER:
        req.append("iss")
    if AUDIENCE:
        req.append("aud")
    return req


def _decode(token: str) -> dict:
    opts = {"require": _required_claims()}
    if JWKS_URL:                                    # 운영: IdP의 공개키(RS256)
        signing_key = _jwks_client(JWKS_URL).get_signing_key_from_jwt(token).key
        return jwt.decode(token, signing_key, algorithms=["RS256"],
                          audience=AUDIENCE or None, issuer=ISSUER or None, options=opts)
    return jwt.decode(token, SECRET, algorithms=["HS256"],   # kind: 개발 토큰
                      audience=AUDIENCE or None, issuer=ISSUER or None, options=opts)


def tenant_from_auth(authorization: str = Header("")) -> Tenant:   # 헤더가 없어도 422가 아니라 401
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer")
    try:
        claims = _decode(authorization[7:])
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "invalid token") from exc
    if not str(claims.get("sub") or "").strip():
        raise HTTPException(401, "invalid token")            # 빈 주체도 거절합니다
    tid = claims.get("tenant_id")
    if not tid:
        raise HTTPException(403, "tenant claim missing")     # fail-closed
    return Tenant(id=str(tid), tier=str(claims.get("tier", "free")),
                  user=str(claims.get("sub", "")))
