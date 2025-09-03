LLM Router / Load Balancer

Overview
- Pluggable providers (ISH, Groq, Cerebras)
- Prompt classification: chat, coding, research
- Heuristic scoring by latency/failure-weighted + provider weights
- Simple RPM rate limiting per provider
- Uses existing ISH client for free premium models (where working)

Quick start
1) Add API keys in .env (GROQ_API_KEY, CEREBRAS_API_KEY). ISH uses free key.
2) python3 -m router.llm_router
3) Integrate into your Flask app:

    from router.llm_router import LLMRouter
    router = LLMRouter()
    text = router.chat(prompt, task="coding")

Extending providers
- Use add_or_update_provider(name, meta) to register new providers dynamically
- Provide base_url and model list or implement discovery

Notes
- This is a minimal but functional baseline. It avoids heavy SDKs and sticks to requests.
- You can wire in Brave/SerpAPI for research augmentation later.
