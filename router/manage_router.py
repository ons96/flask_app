#!/usr/bin/env python3
"""
CLI to manage LLM Router providers/config.
Examples:
  python3 -m router.manage_router list
  python3 -m router.manage_router add --name openrouter --base_url https://openrouter.ai/api/v1 --env_key OPENROUTER_API_KEY --models qwen2.5-coder,meta-llama/llama-3.1-70b-instruct
  python3 -m router.manage_router enable --name groq --on true
"""
import argparse
import json
import os
from typing import Any, Dict

from .llm_router import LLMRouter, _CONFIG_PATH


def load_config() -> Dict[str, Any]:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg: Dict[str, Any]) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list")

    addp = sub.add_parser("add")
    addp.add_argument("--name", required=True)
    addp.add_argument("--base_url")
    addp.add_argument("--env_key")
    addp.add_argument("--models")
    addp.add_argument("--discover", action="store_true")

    enp = sub.add_parser("enable")
    enp.add_argument("--name", required=True)
    enp.add_argument("--on", required=True)

    args = ap.parse_args()

    cfg = load_config()
    if not cfg:
        cfg = {"providers": {}}

    if args.cmd == "list":
        r = LLMRouter()
        print("Providers:")
        for name, meta in r.config.get("providers", {}).items():
            print("-", name, meta)
        print("Discovered models:", r.get_all_models())
        return

    if args.cmd == "add":
        meta: Dict[str, Any] = {"enabled": True, "supports_stream": True, "weight": 1.0, "max_rpm": 60}
        if args.base_url:
            meta["base_url"] = args.base_url
        if args.env_key:
            meta["env_key"] = args.env_key
        if args.models:
            meta["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.discover:
            meta["discover"] = True
        cfg.setdefault("providers", {})[args.name] = meta
        save_config(cfg)
        print(f"Added/updated provider '{args.name}'.")
        return

    if args.cmd == "enable":
        name = args.name
        on = args.on.lower() in ("1", "true", "yes", "y")
        cfg.setdefault("providers", {}).setdefault(name, {})["enabled"] = on
        save_config(cfg)
        print(f"Provider '{name}' enabled={on}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
