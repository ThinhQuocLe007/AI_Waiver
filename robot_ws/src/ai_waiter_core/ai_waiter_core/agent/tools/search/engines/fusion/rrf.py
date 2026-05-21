from typing import List, Tuple
from langchain_core.documents import Document
from ai_waiter_core.schemas.search import SearchResult
from ..scoring import compute_reciprocal_rank
from .base import BaseFusion

class RRFFusion(BaseFusion):
    def fuse(self, 
             bm25_results: List[Tuple[Document, float]], 
             vector_results: List[Tuple[Document, float]], 
             k: int, 
             **kwargs) -> List[SearchResult]:
        
        rrf_k = kwargs.get("rrf_k", 60)
        fusion_scores = {} # page_content_hash -> dict

        # 1. Process BM25
        for rank, (doc, raw_score) in enumerate(bm25_results, 1):
            doc_id = hash(doc.page_content)
            fusion_scores[doc_id] = {
                "doc": doc,
                "score": compute_reciprocal_rank(rank, rrf_k),
                "bm25_score": raw_score,
                "vector_score": 0.0
            }

        # 2. Process Vector
        for rank, (doc, raw_score) in enumerate(vector_results, 1):
            doc_id = hash(doc.page_content)
            rrf_contrib = compute_reciprocal_rank(rank, rrf_k)
            
            if doc_id in fusion_scores:
                fusion_scores[doc_id]["score"] += rrf_contrib
                fusion_scores[doc_id]["vector_score"] = raw_score
            else:
                fusion_scores[doc_id] = {
                    "doc": doc,
                    "score": rrf_contrib,
                    "bm25_score": 0.0,
                    "vector_score": raw_score
                }

        # 3. Format to SearchResult
        final_list = []
        for entry in fusion_scores.values():
            doc = entry["doc"]
            final_list.append(SearchResult(
                document=doc,
                score=entry["score"],
                bm25_score=entry["bm25_score"],
                bm25_normalized=0.0, # Not used in RRF
                vector_score=entry["vector_score"],
                source=doc.metadata.get("source", "unknown"),
                doc_type=doc.metadata.get("type", "unknown")
            ))

        return self._format_results(final_list, k)
