import json
import contextvars
from functools import lru_cache
from app import config

# Token counter for the current request or eval question
_usage = contextvars.ContextVar("usage", default=None)


def start_usage():
    _usage.set({"inputTokens": 0, "outputTokens": 0, "llmCalls": 0})


def get_usage():
    return dict(_usage.get() or {})


@lru_cache(maxsize=1)
def _client():
    return config.bedrock_client()


def chat(prompt, system=None, max_tokens=800):
    kwargs = {
        "modelId": config.CHAT_MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = _client().converse(**kwargs)
    u = _usage.get()
    if u is not None:
        u["inputTokens"] += resp["usage"]["inputTokens"]
        u["outputTokens"] += resp["usage"]["outputTokens"]
        u["llmCalls"] += 1
    return resp["output"]["message"]["content"][0]["text"]


def embed(text):
    body = json.dumps({"inputText": text[:20000], "dimensions": 1024, "normalize": True})
    resp = _client().invoke_model(modelId=config.EMBED_MODEL_ID, body=body)
    return json.loads(resp["body"].read())["embedding"]
