#!/usr/bin/env python3
"""개발용 JWT 발급 (2.5절).

    python3 tools/dev_token.py --tenant demo --tier pro > /tmp/rag-token

운영 토큰은 IdP가 발급합니다. 이 스크립트는 kind에서 게이트웨이를 통과시키기
위한 HS256 토큰만 만듭니다. JWT_SECRET은 게이트웨이와 같은 값을 써야 합니다.
"""
from __future__ import annotations

import argparse
import os
import time

import jwt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--tier", default="pro", choices=["free", "pro", "enterprise"])
    ap.add_argument("--ttl", type=int, default=3600)
    a = ap.parse_args()

    now = int(time.time())
    payload = {
        "iss": os.environ.get("JWT_ISSUER", "rag-dev"),
        "aud": os.environ.get("JWT_AUDIENCE", "rag-api"),
        "sub": f"dev-{a.tenant}",
        "tenant_id": a.tenant,
        "tier": a.tier,
        "iat": now,
        "exp": now + a.ttl,
    }
    token = jwt.encode(payload, os.environ.get("JWT_SECRET", "dev-only-secret"),
                       algorithm="HS256")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
