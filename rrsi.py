#!/usr/bin/env python3
"""RRSI CLI 진입점.

    python3 rrsi.py --domain rag smoke
    python3 rrsi.py --domain rag baseline
    python3 rrsi.py --domain rag run --T 1
    python3 rrsi.py --domain rag status
    RAG_HELDOUT_SPLIT=ood_docs python3 rrsi.py --domain rag heldout --label ood_docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rrsi.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
