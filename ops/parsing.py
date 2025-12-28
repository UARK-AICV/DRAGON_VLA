import json
import ast
import re

# ───────────────────────────── helpers: parsing & IO ─────────────────────────────

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")   # ", }" or ", ]" → "}"

def _between_braces(s: str) -> str:
    """Return substring from first '{' to matching last '}' (fallback: first..last)."""
    start = s.find("{")
    end   = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return s[start:end+1]

def _to_list_of_ints(v):
    """Coerce value to list[int]."""
    if isinstance(v, int):
        return [v]
    if isinstance(v, str):
        # allow "1,2" or "1, 2"
        try:
            return [int(x) for x in v.replace(",", " ").split() if x.strip().isdigit()]
        except Exception:
            return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []

def parse_answer(raw: str) -> dict:
    """
    Robustly parse model output into {snake_key: [int, ...]}.
    Accepts python or json dicts; tolerates trailing commas & code fences.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}

    # 1) strip code fences like ```python ... ```
    s = _FENCE_RE.sub("", raw).strip()

    # 2) extract the dict-ish region
    blob = _between_braces(s)
    if not blob:
        return {}

    # 3) try python literal first (handles trailing commas, single quotes)
    for attempt in range(2):
        try:
            d = ast.literal_eval(blob)
            if not isinstance(d, dict):
                break
            cleaned = {}
            for k, v in d.items():
                k_snake = k.strip().lower().replace("_", " ")
                cleaned[k_snake] = _to_list_of_ints(v)
            return cleaned
        except Exception:
            # one cleanup pass then retry: remove trailing commas
            if attempt == 0:
                blob = _TRAILING_COMMA_RE.sub(r"\1", blob)
            else:
                break

    # 4) fallback: try json after quote normalization and trailing-comma cleanup
    try:
        jblob = blob.replace("'", '"')
        jblob = _TRAILING_COMMA_RE.sub(r"\1", jblob)
        d = json.loads(jblob)
        if isinstance(d, dict):
            cleaned = {}
            for k, v in d.items():
                k_snake = k.strip().lower().replace("_", " ")
                cleaned[k_snake] = _to_list_of_ints(v)
            return cleaned
    except Exception:
        pass

    return {}

