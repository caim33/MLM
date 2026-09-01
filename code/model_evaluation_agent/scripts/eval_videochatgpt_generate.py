#!/usr/bin/env python3
"""Strict catalog facade; never emits proxy or synthetic predictions."""
from runner_support import evaluation_main

if __name__ == "__main__":
    raise SystemExit(evaluation_main(("videochatgpt_lora",)))
