#!/usr/bin/env python3
"""Strict catalog facade; real backends only, unavailable integrations fail closed."""
from runner_support import finetune_main

if __name__ == "__main__":
    raise SystemExit(finetune_main(("qwen3vl_8b_lora", "qwen3vl_4b_lora")))
