#!/usr/bin/env python3
"""Strict catalog facade; real backends only, unavailable integrations fail closed."""
from runner_support import finetune_main

if __name__ == "__main__":
    raise SystemExit(finetune_main(("qwen36_27b_lora",)))
