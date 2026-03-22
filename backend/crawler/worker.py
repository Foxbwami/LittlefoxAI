import re
import requests
from bs4 import BeautifulSoup
from backend.core import config


def fetch_page(url):
    try:
        res = requests.get(url, timeout=config.CRAWL_TIMEOUT, headers={"User-Agent": "LittlefoxAI/1.0"})
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        main = soup.find("main") or soup.find("article")
        paragraphs = main.find_all("p") if main else soup.find_all("p")
        cleaned_paras = []
        for p in paragraphs:
            para = p.get_text(" ", strip=True)
            para = re.sub(r"\[[^\]]+\]", "", para)
            para = re.sub(r"\s+", " ", para).strip()
            if len(para.split()) >= 8:
                cleaned_paras.append(para)
            if len(cleaned_paras) >= 4:
                break
        text = " ".join(cleaned_paras)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            text = soup.get_text(separator=" ", strip=True)
        links = [a.get("href") for a in soup.find_all("a", href=True)]

        return {"url": url, "content": text, "links": links}
    except Exception:
        return None
