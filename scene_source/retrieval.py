"""Small BM25 companion to semantic retrieval; no external search dependency."""
from __future__ import annotations
import math
import re
from collections import Counter

STOP = {"무엇","어떤","어떻게","정말","실제로","관한","관련","설명","주세요","인가요","있나요","하나요","무엇인가요","관계인가요"}
ALIASES = {"블랙홀": "black hole holes", "관측": "observation observations observatory observatories detect",
           "도구": "instrument instruments", "지평선": "event horizon", "웜홀": "wormhole wormholes",
           "중력": "gravity gravitational"}


def terms(text):
    tokens = []
    for word in re.findall(r"[a-zA-Z0-9가-힣]+", text.casefold()):
        if re.fullmatch(r"[가-힣]+", word):
            word = re.sub(r"(에서는|으로는|으로|에서|에게|과|와|은|는|이|가|을|를|의|도)$", "", word)
        if len(word) >= 2 and word not in STOP:
            tokens.append(word)
    return tokens


def bm25_scores(question, texts):
    expanded = question + " " + " ".join(v for k,v in ALIASES.items() if k in question)
    query = set(terms(expanded))
    counters = [Counter(terms(t)) for t in texts]
    n = len(counters)
    average = sum(sum(c.values()) for c in counters) / max(n, 1)
    scores = [0.0] * n
    for term in query:
        freq = sum(term in c for c in counters)
        idf = math.log(1 + (n - freq + .5) / (freq + .5))
        for i, counter in enumerate(counters):
            tf = counter[term]
            if tf:
                length = sum(counter.values())
                scores[i] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / max(average, 1)))
    maximum = max(scores, default=0)
    return [s / maximum if maximum else 0 for s in scores]
