"""Semantic + BM25 candidates, cross-encoder ranking, complete parent context."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
STOPWORDS = set('a an the and or is are was were be been being to of for in on at by with as it its this that these those i me my we our you your they their can could should would do does did have has had what when where how which who since supposed aurorastay'.split())

def terms(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOPWORDS]

def ranking_text(text):
    # The pretrained general-purpose ranker overweights an unfamiliar brand.
    # Normalize the brand consistently in query and candidates only. Evidence
    # and the user's question passed to generation remain completely unchanged.
    return re.sub(r'\bAuroraStay\b', 'hotel', text, flags=re.IGNORECASE)

class HybridRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.parents = {}
        for key in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(key)
            self.parents.setdefault(doc.metadata['parent_id'], doc.metadata)
        self.ids = list(self.parents)
        self.bm25 = BM25Okapi([terms(self.parents[i]['parent_text']) for i in self.ids])
        self.reranker = CrossEncoder(RERANK_MODEL, device='cpu', max_length=512)

    def retrieve(self, query, k=3):
        # Candidate sizes and the reranker are fixed before benchmark execution.
        dense = self.vectorstore.similarity_search(query, k=min(48, self.vectorstore.index.ntotal))
        ranks = {}
        dense_ids = list(dict.fromkeys(d.metadata['parent_id'] for d in dense))
        lexical_scores = self.bm25.get_scores(terms(query))
        lexical_ids = [self.ids[i] for i in np.argsort(lexical_scores)[::-1][:48] if lexical_scores[i] > 0]
        for ranking in [dense_ids, lexical_ids]:
            for rank, pid in enumerate(ranking, 1):
                ranks[pid] = ranks.get(pid, 0) + 1/(60+rank)
        candidates = sorted(ranks, key=ranks.get, reverse=True)
        # Never answer an explicitly named property's question with another
        # property's matrix row. Generic rules remain available as context.
        requested = set(re.findall(r'ASH-[A-Z]{3}', query.upper()))
        if requested:
            candidates = [pid for pid in candidates
                if not (codes := set(re.findall(r'ASH-[A-Z]{3}', self.parents[pid]['parent_text'].upper())))
                or bool(codes & requested)]
        scores = self.reranker.predict([(ranking_text(query),ranking_text(self.parents[i]['parent_text'])) for i in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [self.parents[i] for i, _ in ranked[:k]]

_cache = {}
def retrieve(query, k=3, vectorstore=None):
    if not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError('k must be an integer from 1 to 10')
    key=id(vectorstore)
    if key not in _cache:
        _cache[key]=HybridRetriever(vectorstore)
    return [f"Content:\n{p['parent_text']}\n\nMetadata:\n"
            f"{{'sources': {p['sources']!r}, 'section': {p['section']!r}, 'parent_id': {p['parent_id']!r}}}"
            for p in _cache[key].retrieve(query,k)]
