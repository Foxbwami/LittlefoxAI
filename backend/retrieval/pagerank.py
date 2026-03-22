import numpy as np


def compute_pagerank(pages, d=0.85, iterations=20):
    n = len(pages)
    if n == 0:
        return np.array([])

    ranks = np.ones(n) / n
    link_map = {i: pages[i].get("links", []) for i in range(n)}

    for _ in range(iterations):
        new_ranks = np.ones(n) * (1 - d) / n
        for i, links in link_map.items():
            if not links:
                continue
            share = ranks[i] / len(links)
            for j in range(n):
                if pages[j]["url"] in links:
                    new_ranks[j] += d * share
        ranks = new_ranks
    return ranks
