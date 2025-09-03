#!/usr/bin/env python3
"""
General-purpose LLM Router / Load Balancer
- Pluggable providers (ISH, Groq, Cerebras, OpenAI-compatible)
- Auto model discovery (ISH), optional discovery for others
- Prompt-aware routing (chat vs coding vs research)
- Heuristic and metrics-based provider/model selection (cost, speed, quality, rate limits)
- Simple rate limiting and failure backoff
- Research augmentation (Brave API if available, else DuckDuckGo search)

This module is self-contained and optional-dependency friendly. It avoids heavy SDKs.

Usage (example):
    from router.llm_router import LLMRouter
    router = LLMRouter()
    reply = router.chat(prompt="Explain recursion", task="chat")

Integration with ISH:
    - Uses existing flask_app/ish_client.py (ISHFinalClient)

Note: Actual HTTP calls for Groq/Cerebras are implemented using requests and env vars.
"""
from __future__ import annotations

import os
import time
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

# Local ISH client (copied earlier into flask_app/ish_client.py)
try:
    from ish_client import ISHFinalClient  # type: ignore
    _HAS_ISH = True
except Exception:
    _HAS_ISH = False

_THIS_DIR = os.path.dirname(__file__)
_APP_DIR = os.path.dirname(_THIS_DIR)
_PERF_CSV_PATH = os.path.join(_APP_DIR, "provider_performance.csv")
_CONFIG_PATH = os.path.join(_THIS_DIR, "llm_router_config.json")
_BENCHMARKS_PATH = os.path.join(_THIS_DIR, "benchmarks.json")

# Env
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")  # for research augmentor if available
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Default config
_DEFAULT_CONFIG: Dict[str, Any] = {
    "providers": {
        "ish": {
            "enabled": True,
            "supports_stream": True,
            "supports_research": False,
            "weight": 1.0,
            "max_rpm": 30,
            "models": "auto",  # discovered via ISHFinalClient
            "cost_per_1k": 0.0
        },
        "groq": {
            "enabled": bool(GROQ_API_KEY),
            "supports_stream": True,
            "supports_research": False,
            "weight": 1.0,
            "max_rpm": 60,
            "base_url": "https://api.groq.com/openai/v1",
            "discover": False,
            "models": [
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
                "llama-3.1-8b-instant"
            ],
            "cost_per_1k": 0.0  # set if known
        },
        "cerebras": {
            "enabled": bool(CEREBRAS_API_KEY),
            "supports_stream": True,
            "supports_research": False,
            "weight": 1.0,
            "max_rpm": 60,
            "base_url": "https://api.cerebras.ai/v1",
            "discover": False,
            "models": [
                "llama3.1-8b",
                "llama3.1-70b",
                "llama3.1-8b-instruct",
                "llama3.1-70b-instruct"
            ],
            "cost_per_1k": 0.0
        }
    },
    "routing": {
        "task_profiles": {
            "chat": {
                "prefer_reasoning": False,
                "cost_weight": 0.3,
                "speed_weight": 0.5,
                "quality_weight": 0.7
            },
            "coding": {
                "prefer_reasoning": False,
                "cost_weight": 0.2,
                "speed_weight": 0.6,
                "quality_weight": 0.9,
                "preferred_models": [
                    "qwen2.5-coder",
                    "deepseek-coder",
                    "llama-3.1-70b-instruct",
                    "llama-3.1-8b-instruct"
                ]
            },
            "research": {
                "prefer_reasoning": True,
                "cost_weight": 0.2,
                "speed_weight": 0.4,
                "quality_weight": 1.0
            }
        }
    }
}

# Simple in-memory metrics
@dataclass
class ProviderMetrics:
    rpm: int = 0
    last_reset: float = field(default_factory=time.time)
    latencies: List[float] = field(default_factory=list)
    failures: int = 0

    def can_call(self, max_rpm: int) -> bool:
        now = time.time()
        if now - self.last_reset >= 60:
            self.rpm = 0
            self.last_reset = now
        return self.rpm < max_rpm

    def record(self, latency: float, success: bool):
        self.rpm += 1
        if success:
            self.latencies.append(latency)
            if len(self.latencies) > 50:
                self.latencies = self.latencies[-50:]
        else:
            self.failures += 1

    def avg_latency(self) -> float:
        if not self.latencies:
            return 2.0  # default
        return sum(self.latencies) / len(self.latencies)


class LLMRouter:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or _CONFIG_PATH
        self.config = self._load_config()
        self.metrics: Dict[str, ProviderMetrics] = {name: ProviderMetrics() for name in self.config["providers"].keys()}
        self._ish_client: Optional[ISHFinalClient] = ISHFinalClient() if _HAS_ISH and self.config["providers"].get("ish", {}).get("enabled") else None
        self._lock = threading.Lock()
        self._model_cache: Dict[str, List[str]] = {}
        self._benchmarks: Dict[str, float] = self._load_benchmarks()
        # prime caches
        try:
            _ = self.get_all_models()
        except Exception:
            pass

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # merge defaults shallowly
                merged = json.loads(json.dumps(_DEFAULT_CONFIG))
                for k, v in cfg.items():
                    merged[k] = v
                return merged
        except Exception:
            pass
        return json.loads(json.dumps(_DEFAULT_CONFIG))

    def _save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def _load_benchmarks(self) -> Dict[str, float]:
        if os.path.exists(_BENCHMARKS_PATH):
            try:
                with open(_BENCHMARKS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    # -------- Model Discovery --------
    def get_all_models(self) -> Dict[str, List[str]]:
        if self._model_cache:
            return self._model_cache
        models: Dict[str, List[str]] = {}
        providers = self.config.get("providers", {})
        # ISH
        if providers.get("ish", {}).get("enabled") and self._ish_client:
            try:
                w = self._ish_client.get_working_models()  # List[Dict]
                models["ish"] = [m["id"] for m in w]
            except Exception:
                models["ish"] = []
        # Generic OpenAI-compatible discovery
        for prov_name in [p for p in providers.keys() if p not in ("ish",)]:
            pconf = providers[prov_name]
            if not pconf.get("enabled"):
                continue
            if pconf.get("discover") and pconf.get("base_url"):
                try:
                    discovered = self._discover_models_via_openai(pconf)
                    models[prov_name] = discovered
                except Exception:
                    models[prov_name] = pconf.get("models", [])
            else:
                models[prov_name] = pconf.get("models", [])
        self._model_cache = models
        return models

    def _discover_models_via_openai(self, pconf: Dict[str, Any]) -> List[str]:
        base = pconf.get("base_url").rstrip("/")
        key = os.getenv(pconf.get("env_key", "")) or pconf.get("api_key")
        if not key:
            return pconf.get("models", [])
        url = f"{base}/models"
        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: List[str] = []
        for item in data.get("data", []):
            mid = item.get("id")
            if mid:
                out.append(mid)
        return out

    # -------- Prompt Classification --------
    def classify(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ["http://", "https://", "search", "research", "sources", "citations", "latest", "news"]):
            return "research"
        if any(k in p for k in ["code", "bug", "function", "class", "traceback", "compile", "error", "write a script", "regex", "sql", "python", "javascript", "typescript", "java", "c#", "go", "rust"]):
            return "coding"
        # complexity heuristic
        if len(prompt) > 600 or len(re.findall(r"\n", prompt)) > 15:
            return "research" if any(k in p for k in ["analyze", "compare", "survey", "paper", "dataset"]) else "chat"
        return "chat"

    # -------- Scoring helpers --------
    def _quality_score(self, model: str) -> float:
        # Normalize 0..1
        if model in self._benchmarks:
            val = self._benchmarks[model]
            return max(0.0, min(1.0, float(val)))
        # heuristic defaults
        if any(k in model.lower() for k in ["reasoning", "r1", "o4", "qwen2.5-coder", "deepseek-coder"]):
            return 0.85
        if any(k in model.lower() for k in ["llama", "mixtral", "gemma"]):
            return 0.7
        return 0.6

    def _cost_score(self, provider: str, model: str) -> float:
        # lower cost -> higher score (0..1)
        pconf = self.config["providers"].get(provider, {})
        cost =  pconf.get("cost_per_1k", 0.0)
        # map cost to score; assume 0..1$ -> 1..0.2
        if cost <= 0:
            return 1.0
        return max(0.2, 1.0 / (1.0 + cost))

    # -------- Provider/Model Selection --------
    def choose(self, prompt: str, task: Optional[str] = None) -> Tuple[str, str]:
        task = task or self.classify(prompt)
        profile = self.config["routing"]["task_profiles"].get(task, self.config["routing"]["task_profiles"]["chat"]) 
        all_models = self.get_all_models()
        candidates: List[Tuple[str, str]] = []
        # Use preferred coding models when coding
        if task == "coding":
            preferred = [p.lower() for p in profile.get("preferred_models", [])]
            for prov, mlist in all_models.items():
                for m in mlist:
                    if any(pref in m.lower() for pref in preferred):
                        candidates.append((prov, m))
        # Fallback to all
        if not candidates:
            for prov, mlist in all_models.items():
                for m in mlist:
                    candidates.append((prov, m))
        if not candidates:
            return ("ish", "deepseek-chat")
        # Score
        def score(pair: Tuple[str, str]) -> float:
            prov, model = pair
            pm = self.metrics.get(prov)
            lat = pm.avg_latency() if pm else 2.0
            fail_penalty = 1.0 + (pm.failures * 0.05 if pm else 0)
            weight = self.config["providers"].get(prov, {}).get("weight", 1.0)
            q = self._quality_score(model)
            c = self._cost_score(prov, model)
            return (
                profile.get("quality_weight", 0.7) * q +
                profile.get("speed_weight", 0.5) * (1.0 / (lat * fail_penalty)) +
                profile.get("cost_weight", 0.3) * c
            ) * weight
        candidates.sort(key=score, reverse=True)
        # Rate limit aware selection
        for prov, model in candidates:
            meta = self.config["providers"][prov]
            if not meta.get("enabled"):
                continue
            pm = self.metrics[prov]
            if pm.can_call(meta.get("max_rpm", 60)):
                return prov, model
        # If all throttled, pick the best and proceed
        return candidates[0]

    # -------- Chat API --------
    def chat(self, prompt: str, task: Optional[str] = None, system: Optional[str] = None, stream: bool = False, max_tokens: int = 1500, temperature: float = 0.7) -> str:
        provider, model = self.choose(prompt, task)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        start = time.time()
        try:
            text = self._call_provider(provider, model, messages, stream=stream, max_tokens=max_tokens, temperature=temperature)
            self.metrics[provider].record(time.time() - start, success=True)
            return text
        except Exception:
            self.metrics[provider].record(time.time() - start, success=False)
            # simple backoff: try others quickly
            alt = self.get_all_models()
            for prov, mlist in alt.items():
                if prov == provider:
                    continue
                if not self.config["providers"].get(prov, {}).get("enabled"):
                    continue
                if not mlist:
                    continue
                try:
                    return self._call_provider(prov, mlist[0], messages, stream=stream, max_tokens=max_tokens, temperature=temperature)
                except Exception:
                    continue
            raise

    def chat_with_research(self, query: str, max_results: int = 4, tokens: int = 1200) -> str:
        """Do web search (Brave if available, else DuckDuckGo), fetch pages, and synthesize with citations."""
        results = self._search(query, max_results=max_results)
        docs: List[Tuple[str, str]] = []  # (title, text)
        for r in results:
            url = r.get("url") or r.get("link")
            title = r.get("title") or url
            if not url:
                continue
            text = self._fetch_text(url)
            if text:
                docs.append((title, text[:6000]))  # cap per doc
        if not docs:
            return self.chat(query, task="research", max_tokens=tokens)
        # Build synthesis prompt
        citations = "\n\n".join([f"- {t}: {results[i].get('url') or results[i].get('link')}" for i, (t, _) in enumerate(docs)])
        context = "\n\n".join([f"Source {i+1} ({t}):\n{txt}" for i, (t, txt) in enumerate(docs)])
        sysmsg = (
            "You are a research assistant. Read the provided sources and produce a concise, accurate answer. "
            "Cite sources inline as [S1], [S2], etc. Provide a short final references list with URLs."
        )
        user = f"Question: {query}\n\nContext Sources (S1..S{len(docs)}):\n{context}\n\nProvide answer with inline citations and a References section."
        answer = self.chat(user, task="research", system=sysmsg, max_tokens=tokens)
        return answer + "\n\nReferences:\n" + citations

    # -------- Provider Calls --------
    def _call_provider(self, provider: str, model: str, messages: List[Dict[str, str]], stream: bool = False, max_tokens: int = 1500, temperature: float = 0.7) -> str:
        if provider == "ish" and self._ish_client:
            # Prefer robust continuation path
            try:
                result = self._ish_client.chat_completion_with_continuation(model=model, messages=messages, max_tokens=max_tokens)  # type: ignore[arg-type]
                if isinstance(result, dict):
                    text = result.get("content") or result.get("text") or ""
                else:
                    text = str(result)
                if self._appears_truncated(text):
                    # ask to continue once
                    cont_msgs = messages + [{"role": "user", "content": "Continue."}]
                    extra = self._ish_client.chat_completion_non_streaming(model=model, messages=cont_msgs, max_tokens=max_tokens)  # type: ignore[arg-type]
                    text += "\n" + (extra if isinstance(extra, str) else str(extra))
                return text
            except Exception:
                # fallback to non-streaming
                result = self._ish_client.chat_completion_non_streaming(model=model, messages=messages, max_tokens=max_tokens)  # type: ignore[arg-type]
                if isinstance(result, dict):
                    return result.get("content") or result.get("text") or ""
                return str(result)

        if provider == "groq" and GROQ_API_KEY:
            url = self.config["providers"]["groq"]["base_url"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if provider == "cerebras" and CEREBRAS_API_KEY:
            url = self.config["providers"]["cerebras"]["base_url"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Generic fallback: echo-like behavior
        return "[Router] No provider available or configured. Please add API keys in .env or router config."

    # -------- Config Management --------
    def add_or_update_provider(self, name: str, meta: Dict[str, Any]) -> None:
        with self._lock:
            self.config.setdefault("providers", {})[name] = meta
            self._model_cache = {}
            self._save_config()

    def get_models_for_task(self, task: str) -> List[Tuple[str, str]]:
        """Return (provider, model) pairs likely good for the task."""
        pairs: List[Tuple[str, str]] = []
        models = self.get_all_models()
        profile = self.config["routing"]["task_profiles"].get(task, {})
        preferred = [p.lower() for p in profile.get("preferred_models", [])]
        for prov, mlist in models.items():
            for m in mlist:
                if not preferred or any(p in m.lower() for p in preferred):
                    pairs.append((prov, m))
        return pairs

    # -------- Helpers --------
    def _appears_truncated(self, text: str) -> bool:
        t = text.strip()
        if len(t) < 100:
            return False
        if t.endswith(("...", "..", "—", "-", ":", ";")):
            return True
        # likely complete if ends with full stop
        if t.endswith( (".", "!", "?") ):
            return False
        return True

    def _search(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        # 1) Tavily first (good relevance, has generous free tier)
        if TAVILY_API_KEY:
            try:
                url = "https://api.tavily.com/search"
                payload = {"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}
                r = requests.post(url, json=payload, timeout=30)
                if r.ok:
                    data = r.json()
                    results = data.get("results", [])
                    return [{"title": it.get("title"), "url": it.get("url") } for it in results][:max_results]
            except Exception:
                pass
        # 2) Brave
        if BRAVE_API_KEY:
            try:
                url = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": BRAVE_API_KEY}
                params = {"q": query, "count": max_results}
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.ok:
                    data = r.json()
                    results = data.get("web", {}).get("results", [])
                    out = [{"title": it.get("title"), "url": it.get("url") } for it in results]
                    return out[:max_results]
            except Exception:
                pass
        # 3) SerpAPI (if available)
        if SERPAPI_API_KEY:
            try:
                url = "https://serpapi.com/search.json"
                params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": max_results}
                r = requests.get(url, params=params, timeout=30)
                if r.ok:
                    data = r.json()
                    items = data.get("organic_results", [])
                    out = [{"title": it.get("title"), "url": it.get("link") } for it in items]
                    return out[:max_results]
            except Exception:
                pass
        # 4) DuckDuckGo (requires duckduckgo_search package if installed)
        try:
            from duckduckgo_search import DDGS  # type: ignore
            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append({"title": r.get("title"), "url": r.get("href") or r.get("url")})
            return out
        except Exception:
            return []

    def _fetch_text(self, url: str, max_bytes: int = 300_000) -> str:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if not r.ok:
                return ""
            content = r.text
            if len(content) > max_bytes:
                content = content[:max_bytes]
            # strip HTML tags
            text = re.sub(r"<script[\s\S]*?</script>", " ", content, flags=re.I)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception:
            return ""


if __name__ == "__main__":
    router = LLMRouter()
    print("Providers:", list(router.config.get("providers", {}).keys()))
    print("Discovered models:", router.get_all_models())
    print("Classification demo:")
    for p in [
        "Fix this Python function to avoid IndexError",
        "Summarize the latest news on SpaceX Starship and cite sources",
        "Explain memoization"
    ]:
        print(p, "->", router.classify(p))
    print("Chat demo:")
    print(router.chat("Hello! Please reply with: router working", task="chat", max_tokens=50))
    print("Research demo:")
    print(router.chat_with_research("Latest announcements from OpenAI about GPT-5", max_results=2, tokens=500))
