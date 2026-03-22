from multiprocessing import Pool

from backend.crawler.worker import fetch_page
from backend.core import config


def crawl_distributed(urls):
    with Pool(config.CRAWL_WORKERS) as p:
        results = p.map(fetch_page, urls)
    return [r for r in results if r]
