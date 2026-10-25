"""Convenience entry point for the DevOps Repository Auditing Agent."""

from __future__ import annotations

import sys

from agent import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
