"""
Wrapper around app_enhanced_context.py that:
- Forces a fresh scrape of provider performance data on startup so priority models can change each run
- Patches web search with re-ranking and filtering to improve relevance

Run this file instead of app_enhanced_context.py.
"""
import os
import re
import math
import time
from typing import List, Dict

# Import the original app module (unchanged)
import app_enhanced_context as base

# --------------- Helper: Simple lexical relevance scoring ---------------
_STOPWORDS = set(
    "a an and are as at be by for from has have i in is it its of on or that the to was were will with you your".split()
)

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    # keep words and numbers
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _tf(text: str) -> Dict[str, float]:
    tokens = _tokenize(text)
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = float(len(tokens)) or 1.0
    return {t: c / total for t, c in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    # cosine over tf weights (idf-less but effective for short snippets)
    shared = set(a.keys()) & set(b.keys())
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def _re_rank_results(results: List[Dict[str, str]], query: str, top_k: int = 5) -> List[Dict[str, str]]:
    if not results:
        return []
    q_tf = _tf(query)
    scored = []
    seen_urls = set()
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("link") or r.get("url") or ""
        # Skip obvious unrelated or duplicate URLs
        if not title and not snippet:
            continue
        if url and url in seen_urls:
            continue
        text = f"{title}\n{snippet}"
        score = _cosine(_tf(text), q_tf)
        scored.append((score, r))
        if url:
            seen_urls.add(url)
    # sort by score desc
    scored.sort(key=lambda x: x[0], reverse=True)
    # filter out very low scoring entries
    filtered = [r for s, r in scored if s >= 0.02]  # tiny threshold to drop noise
    if not filtered:
        # if everything filtered, keep top few anyway
        filtered = [r for _, r in scored[:top_k]]
    return filtered[:top_k]


def _refine_query(query: str) -> str:
    # Heuristic refinement: keep top non-stopword tokens, add quotes around multiword terms
    tokens = _tokenize(query)
    # prefer longer tokens
    tokens.sort(key=len, reverse=True)
    top = tokens[:6]
    refined = " ".join(top)
    return refined or query


# --------------- Patch: Better web search ---------------
_original_perform_web_search = base.perform_web_search


def perform_web_search_patched(query: str, max_results: int = 5):
    # First try original pipeline
    results = []
    try:
        results = _original_perform_web_search(query, max_results=max_results)
    except Exception as e:
        print(f"--- Patched web search: original pipeline failed: {e} ---")
        results = []

    # Re-rank and filter for relevance
    ranked = _re_rank_results(results, query, top_k=max_results)

    # If still weak relevance (e.g., all scores low), try a refined query once
    if not ranked or len(ranked) < max(1, max_results // 3):
        refined = _refine_query(query)
        if refined and refined != query:
            print(f"--- Patched web search: refining query -> {refined} ---")
            try:
                alt = _original_perform_web_search(refined, max_results=max_results)
                ranked2 = _re_rank_results(alt, query, top_k=max_results)
                if ranked2:
                    ranked = ranked2
            except Exception as e:
                print(f"--- Patched web search: refined search failed: {e} ---")

    return ranked


# Monkey-patch into base module
base.perform_web_search = perform_web_search_patched


# --------------- Force fresh provider performance scrape on startup ---------------
try:
    print("--- [REFRESHED WRAPPER] Forcing fresh performance scrape using enhanced scraper ---")
    scraped = []
    try:
        from scraper_update import scrape_provider_performance
        scraped = scrape_provider_performance()
        print(f"--- [REFRESHED WRAPPER] Enhanced scraper returned {len(scraped)} entries ---")
    except Exception as e:
        print(f"--- [REFRESHED WRAPPER] Enhanced scraper failed: {e} ---")

    if scraped:
        base.save_performance_to_csv(scraped, base.PERFORMANCE_CSV_PATH)
        base.PROVIDER_PERFORMANCE_CACHE = scraped
        print(f"--- [REFRESHED WRAPPER] Refreshed {len(scraped)} performance entries ---")
    else:
        print("--- [REFRESHED WRAPPER] Scrape returned no data; keeping existing cache ---")
    
    # Add missing user-reported models manually
    print("--- [REFRESHED WRAPPER] Adding missing user-reported models ---")
    missing_models = [
        {
            'provider_name_scraped': 'Cerebras',
            'model_name_scraped': 'Qwen3 235B 2507 (Non-reasoning)',
            'context_window': '131k',
            'intelligence_index': 45,
            'response_time_s': 0.68,
            'tokens_per_s': 1258.2,
            'source_url': 'https://artificialanalysis.ai/models/qwen3-235b-a22b-instruct-2507',
            'last_updated_utc': '2025-01-03 12:00:00',
            'is_free_source': 'true'
        },
        {
            'provider_name_scraped': 'OpenAI Compatible',
            'model_name_scraped': 'gpt-oss-120B (high)',
            'context_window': '131k',
            'intelligence_index': 58,
            'response_time_s': 1.05,
            'tokens_per_s': 3202.6,
            'source_url': 'https://artificialanalysis.ai/models/gpt-oss-120b',
            'last_updated_utc': '2025-01-03 12:00:00',
            'is_free_source': 'true'
        },
        {
            'provider_name_scraped': 'Google (AI Studio)',
            'model_name_scraped': 'Gemini 2.5 Flash-Lite (AI Studio)',
            'context_window': '1m',
            'intelligence_index': 30,
            'response_time_s': 1.61,
            'tokens_per_s': 360,
            'source_url': 'https://artificialanalysis.ai/models/gemini-2-5-flash-lite',
            'last_updated_utc': '2025-01-03 12:00:00',
            'is_free_source': 'true'
        }
    ]
    
    # Add missing models to the cache
    for missing_model in missing_models:
        base.PROVIDER_PERFORMANCE_CACHE.append(missing_model)
    
    print(f"--- [REFRESHED WRAPPER] Added {len(missing_models)} missing models ---")

    # Rebuild caches that depend on performance data regardless
    base.initialize_model_cache()
    print(f"--- [REFRESHED WRAPPER] Rebuilt model cache ({len(base.CACHED_AVAILABLE_MODELS_SORTED_LIST)} models) ---")
except Exception as e:
    print(f"--- [REFRESHED WRAPPER] Error during forced scrape: {e} ---")
    import traceback
    print(f"--- [REFRESHED WRAPPER] Full traceback: {traceback.format_exc()} ---")
    # Ensure cache is initialized even if scrape failed
    try:
        base.initialize_model_cache()
        print(f"--- [REFRESHED WRAPPER] Fallback cache initialization: {len(base.CACHED_AVAILABLE_MODELS_SORTED_LIST)} models ---")
    except Exception as init_error:
        print(f"--- [REFRESHED WRAPPER] Cache initialization also failed: {init_error} ---")


# --------------- Expose the same Flask app instance ---------------
app = base.app


if __name__ == "__main__":
    # Optionally show top priority models after rebuild
    try:
        sm = base.CACHED_AVAILABLE_MODELS_SORTED_LIST
        print(f"--- [REFRESHED WRAPPER] Available models count: {len(sm)} ---")
        for i, row in enumerate(sm[:8]):
            print(f"{i+1:02d}. {row}")
    except Exception:
        pass

    print("--- [REFRESHED WRAPPER] Starting Flask server ---")
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
