import os
import time
import requests
import urllib.parse
from backend.core import config

HEADERS = {"User-Agent": "LittlefoxAI/1.0"}


def _timeout():
    return getattr(config, "WEB_TIMEOUT", 6)


def _duckduckgo(query, top_k):
    q = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
    try:
        res = requests.get(url, timeout=_timeout(), headers=HEADERS)
        res.raise_for_status()
        data = res.json()
    except Exception:
        return []

    results = []
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading", "DuckDuckGo"),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", "")[: config.WEB_MAX_CHARS],
                "source": "duckduckgo",
            }
        )

    related = data.get("RelatedTopics", [])
    for item in related:
        if isinstance(item, dict) and item.get("Text"):
            results.append(
                {
                    "title": item.get("Text", "")[:80],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", "")[: config.WEB_MAX_CHARS],
                    "source": "duckduckgo",
                }
            )
        if len(results) >= top_k:
            break

    return results[:top_k]


def _duckduckgo_html(query, top_k):
    q = urllib.parse.quote(query)
    url = f"https://r.jina.ai/http://duckduckgo.com/html/?q={q}"
    try:
        res = requests.get(url, timeout=_timeout() + 4, headers=HEADERS)
        res.raise_for_status()
        text = res.text
    except Exception:
        return []

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("## ["):
            continue
        # Format: ## [Title](url)
        try:
            title_part = line.split("## [", 1)[1]
            title, rest = title_part.split("](", 1)
            url_part = rest.split(")", 1)[0]
            link = url_part
            # Decode DuckDuckGo redirect to original URL when possible
            if "duckduckgo.com/l/?" in link and "uddg=" in link:
                parsed = urllib.parse.urlparse(link)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    link = urllib.parse.unquote(qs["uddg"][0])
            if "duckduckgo.com/y.js" in link or "ad_domain" in link:
                continue
            results.append(
                {
                    "title": title[:120],
                    "url": link,
                    "snippet": title[: config.WEB_MAX_CHARS],
                    "source": "duckduckgo_html",
                }
            )
        except Exception:
            continue
        if len(results) >= top_k:
            break
    return results[:top_k]


def _bing(query, top_k):
    key = os.getenv("BING_API_KEY")
    if not key:
        return []
    url = "https://api.bing.microsoft.com/v7.0/search"
    try:
        res = requests.get(
            url,
            params={"q": query, "count": top_k},
            headers={**HEADERS, "Ocp-Apim-Subscription-Key": key},
            timeout=_timeout(),
        )
        res.raise_for_status()
        data = res.json()
    except Exception:
        return []

    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append(
            {
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", "")[: config.WEB_MAX_CHARS],
                "source": "bing",
            }
        )
    return results[:top_k]


def _google(query, top_k):
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    try:
        res = requests.get(
            url,
            params={"q": query, "key": key, "cx": cx, "num": top_k},
            timeout=_timeout(),
            headers=HEADERS,
        )
        res.raise_for_status()
        data = res.json()
    except Exception:
        return []

    results = []
    for item in data.get("items", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")[: config.WEB_MAX_CHARS],
                "source": "google",
            }
        )
    return results[:top_k]


def _wikipedia(query, top_k):
    url = "https://en.wikipedia.org/w/api.php"
    try:
        res = requests.get(
            url,
            params={"action": "opensearch", "search": query, "limit": top_k, "namespace": 0, "format": "json"},
            timeout=_timeout(),
            headers=HEADERS,
        )
        res.raise_for_status()
        data = res.json()
    except Exception:
        return []

    results = []
    if len(data) >= 2:
        titles = data[1]
        links = data[3] if len(data) > 3 else []
        for idx, title in enumerate(titles[:top_k]):
            link = links[idx] if idx < len(links) else ""
            summary = ""
            try:
                sres = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}",
                    timeout=_timeout(),
                    headers=HEADERS,
                )
                if sres.status_code == 200:
                    sdata = sres.json()
                    summary = sdata.get("extract", "")
            except Exception:
                summary = ""
            results.append(
                {
                    "title": title,
                    "url": link,
                    "snippet": summary[: config.WEB_MAX_CHARS],
                    "source": "wikipedia",
                }
            )
    return results[:top_k]


def search_web(query, top_k=None):
    top_k = top_k or config.WEB_SEARCH_TOP_K
    total_budget = getattr(config, "WEB_TOTAL_TIMEOUT", 12)
    start = time.time()
    results = []
    results.extend(_bing(query, top_k))
    if time.time() - start > total_budget:
        return results[:top_k]
    results.extend(_google(query, top_k))
    if time.time() - start > total_budget:
        return results[:top_k]
    results.extend(_wikipedia(query, top_k))
    if time.time() - start > total_budget:
        return results[:top_k]
    results.extend(_duckduckgo(query, top_k))

    if not results:
        results.extend(_duckduckgo_html(query, top_k))

    if not results:
        simple = _simplify_query(query)
        if simple and simple != query:
            if time.time() - start <= total_budget:
                results.extend(_bing(simple, top_k))
            if time.time() - start <= total_budget:
                results.extend(_google(simple, top_k))
            if time.time() - start <= total_budget:
                results.extend(_wikipedia(simple, top_k))
            if time.time() - start <= total_budget:
                results.extend(_duckduckgo(simple, top_k))
            if time.time() - start <= total_budget and not results:
                results.extend(_duckduckgo_html(simple, top_k))

    seen = set()
    deduped = []
    for r in results:
        key = r.get("url") or r.get("snippet")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped[:top_k]


def _simplify_query(query):
    stop = {
        "summarize", "explain", "define", "describe", "importance", "in", "the", "of",
        "for", "and", "to", "a", "an", "academic", "research", "essay", "write",
        "two", "three", "sentence", "sentences",
    }
    words = [w.strip(".,!?()[]{}:;\"'").lower() for w in query.split()]
    filtered = [w for w in words if w and w not in stop and not w.isdigit()]
    return " ".join(filtered)
