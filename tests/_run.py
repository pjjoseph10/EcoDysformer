"""Tiny no-dependency test runner so the suites run without pytest installed.

Each test module calls ``run_module_tests(globals())`` in its ``__main__`` block;
this collects every ``test_*`` callable, runs it, and reports pass/fail. Under
pytest the same ``test_*`` functions are collected normally.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_module_tests(ns: dict) -> int:
    tests = {n: f for n, f in ns.items()
             if n.startswith("test_") and callable(f)}
    failed = 0
    for name, fn in sorted(tests.items()):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0
