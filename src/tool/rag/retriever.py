import os
import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from config.config_loader import ConfigLoader

from src.tool.rag.bm25 import BM25Index
from src.tool.rag.vectorstore import VectorStore
from src.tool.rag.document_loader import DocumentLoader

@dataclass
class SearchResult:
    """Data class for search results"""
    document: Document
    score: float
    bm25_score: float  # Raw BM25 score
    bm25_normalized: float  # Normalized BM25 score (0-1)
    vector_score: float  # Vector/RAG score (0-1)
    source: str
    doc_type: str


class RetrieverManager:
    """Main class that orchestrates all retrieval operations"""
    
    def __init__(self, config, score_threshold: float = 0.3):
        self.config = config 
        self.data_loader = DocumentLoader(self.config)
        self.score_threshold = score_threshold  
        
        vector_db_path = self.config.faiss_path
        bm25_db_path = self.config.bm25_path 
        
        self.vector_store = VectorStore(vector_db_path)
        self.bm25_index = BM25Index(bm25_db_path, self.data_loader)
        
        self.documents = []
        self.is_ready = False
        self.bm25_mean_score = 0.0 # use for sigmoid normalization
    
    def build_database(self) -> bool:
        """Build complete database (vector + BM25) from data sources"""
        print('[ INFO ] Building Complete Database (Vector + BM25)')
        
        # Load all documents
        self.documents = []
        
        # Load menu data
        menu_docs = self.data_loader.load_menu_data()
        self.documents.extend(menu_docs)
        
        # Load restaurant info
        restaurant_docs = self.data_loader.load_restaurant_info()
        self.documents.extend(restaurant_docs)
        
        if not self.documents:
            print("❌ No documents found! Check your data files.")
            return False
        
        # Build vector store
        if not self.vector_store.build(self.documents):
            return False
        
        # Build BM25 index
        if not self.bm25_index.build(self.documents):
            return False
        
        # Print summary
        self._print_database_summary()
        self.is_ready = True
        return True
    
    def load_database(self) -> bool:
        """Load existing database"""
        print('[ INFO ] Loading existing database...')
        
        vector_loaded = self.vector_store.load()
        bm25_loaded = self.bm25_index.load()
        
        if vector_loaded and bm25_loaded:
            self.documents = self.bm25_index.documents
            self.is_ready = True
            print(' Database loaded successfully')
            return True
        else:
            print('❌ Failed to load database')
            return False
    
    def _sigmoid_normalize(self, score: float, mean: float = 0.0, scale: float = 1.0) -> float:
        """
        Normalize BM25 score to 0-1 range using sigmoid function
        
        Formula: sigmoid(x) = 1 / (1 + e^(-k*(x-mean)))
        
        Args:
            score: BM25 score to normalize
            mean: Mean/center point of the sigmoid (shifts the curve)
            scale: Steepness of the sigmoid curve (default 1.0)
        
        Returns:
            Normalized score in 0-1 range
        """
        try:
            # Clamp exponent to prevent overflow
            exponent = -scale * (score - mean)
            exponent = max(min(exponent, 500), -500)  # Prevent overflow/underflow
            
            sigmoid = 1.0 / (1.0 + math.exp(exponent))
            return sigmoid
        except Exception as e:
            print(f"⚠️ Error in sigmoid normalization: {e}")
            return 0.5
    
    def _calculate_combined_score(self, bm25_score: float, vector_score: float, 
                                 bm25_weight: float = 0.6, vector_weight: float = 0.4) -> float:
        """
        Combine BM25 and vector scores with weighted average
        
        BM25 is normalized using sigmoid, vector score is already 0-1
        """
        # Normalize BM25 score using sigmoid
        normalized_bm25 = self._sigmoid_normalize(bm25_score, mean=self.bm25_mean_score, scale=1.0)
        
        return (normalized_bm25 * bm25_weight) + (vector_score * vector_weight)
    
    def hybrid_search(self, query: str, k: int = 5, threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Perform hybrid search using both BM25 and vector search with scores
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum score threshold (0-1). If None, uses self.score_threshold
        """
        if not self.is_ready:
            print("⚠️ Retriever not initialized. Call initialize() first.")
            return []
        
        if threshold is None:
            threshold = self.score_threshold
        
        print(f"🔍 Hybrid searching: '{query}' (threshold: {threshold:.2f})")
        
        # BM25 search (get more results for better normalization)
        bm25_results = self.bm25_index.search(query, k * 2)
        
        # Calculate mean BM25 score for sigmoid normalization
        if bm25_results:
            scores = [score for _, score in bm25_results]
            self.bm25_mean_score = sum(scores) / len(scores)
            print(f"📊 BM25 Mean Score: {self.bm25_mean_score:.4f}")
        
        # Vector search WITH scores (FAISS returns distances: lower = more similar)
        raw_vector_results = self.vector_store.search_with_scores(query, k * 2)

        # Normalize vector distances to similarity scores in [0, 1], higher = better
        vector_results_with_scores = []
        if raw_vector_results:
            distances = [score for _, score in raw_vector_results]
            min_d, max_d = min(distances), max(distances)
            range_d = max_d - min_d if max_d != min_d else 1.0

            for doc, dist in raw_vector_results:
                # Map distance to similarity: 1 - normalized_distance
                norm_sim = 1.0 - ((dist - min_d) / range_d)
                vector_results_with_scores.append((doc, norm_sim))
        
        # Combine results with scores
        combined = self._combine_results(bm25_results, vector_results_with_scores, k * 2)
        
        # Filter by threshold
        filtered_results = [
            result for result in combined 
            if result.score >= threshold
        ]
        
        # Sort by score (descending) and limit to k results
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        filtered_results = filtered_results[:k]
        
        # Print results with scores
        self._print_results_with_scores(filtered_results, threshold)
        
        print(f"Found {len(filtered_results)} results (above threshold {threshold:.2f})\n")
        return filtered_results
    
    def _combine_results(self, bm25_results: List[Tuple[Document, float]], 
                        vector_results: List[Tuple[Document, float]], k: int) -> List[SearchResult]:
        """Combine BM25 and vector search results with sigmoid-normalized scores"""
        combined = []
        seen_content = set()
        doc_to_vector_score = {}
        
        # Map documents to actual vector scores
        for doc, vector_score in vector_results:
            content_hash = hash(doc.page_content)
            doc_to_vector_score[content_hash] = vector_score
        
        # Add BM25 results first (better for keyword matching)
        for doc, bm25_score in bm25_results:
            if len(combined) >= k:
                break
            
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                # Get vector score if available, otherwise use 0
                vector_score = doc_to_vector_score.get(content_hash, 0.0)
                
                # Normalize BM25 using sigmoid
                bm25_normalized = self._sigmoid_normalize(bm25_score, mean=self.bm25_mean_score, scale=1.0)
                
                # Calculate combined score
                combined_score = (bm25_normalized * 0.6) + (vector_score * 0.4)
                
                result = SearchResult(
                    document=doc,
                    score=combined_score,
                    bm25_score=bm25_score,
                    bm25_normalized=bm25_normalized,
                    vector_score=vector_score,
                    source=doc.metadata.get('source', 'unknown'),
                    doc_type=doc.metadata.get('type', 'unknown')
                )
                combined.append(result)
                seen_content.add(content_hash)
        
        return combined
    
    def _print_results_with_scores(self, results: List[SearchResult], threshold: float = 0.0):
        """Print search results with separate BM25 and RAG scores (without icons)"""
        if not results:
            print("No results found above threshold.")
            return
        
        print("\nSearch Results (BM25 | RAG | Combined):")
        print("-" * 110)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.document.page_content[:70]}...")
            print(f"   BM25: {result.bm25_score:6.2f} → {result.bm25_normalized:.4f} | "
                  f"RAG: {result.vector_score:.4f} | Combined: {result.score:.4f}")
            print(f"   Source: {result.source} | Type: {result.doc_type}")
            print()
        
        print("-" * 110)
    
    def _print_database_summary(self):
        """Print database statistics"""
        menu_count = len([d for d in self.documents if d.metadata.get('source') == 'menu'])
        restaurant_count = len([d for d in self.documents if d.metadata.get('source') == 'restaurant_info'])
        
        print(f"""
                📊 Database Summary:
                • Total documents: {len(self.documents)}
                • Menu items: {menu_count}
                • Restaurant info: {restaurant_count}
                • Vector store: ✅ Ready
                • BM25 index: ✅ Ready
                • Score threshold: {self.score_threshold}
                • Normalization: Sigmoid
                        """)
    
    def set_threshold(self, threshold: float):
        """Update score threshold (0-1)"""
        if 0 <= threshold <= 1:
            self.score_threshold = threshold
            print(f"✅ Threshold updated to {threshold}")
        else:
            print(f"❌ Threshold must be between 0 and 1")


# Convenience objects 
config = ConfigLoader()
_retriever_manager = None
def _get_retriever_manager(score_threshold: float = 0.3) -> RetrieverManager:
    """
    Lazily create and initialize a single RetrieverManager instance.
    The database is loaded only once per process.
    """
    global _retriever_manager

    if _retriever_manager is None:
        _retriever_manager = RetrieverManager(config, score_threshold=score_threshold)
        if not _retriever_manager.load_database():
            print("No existing database found. Call build_vector_db() first.")
    else:
        # Update threshold if needed
        _retriever_manager.set_threshold(score_threshold)

    return _retriever_manager


def build_vector_db() -> bool:
    """Build vector + BM25 database once and keep it in memory."""
    manager = _get_retriever_manager()
    return manager.build_database()


def get_retriever():
    """Get LangChain retriever object (vector-only)."""
    manager = _get_retriever_manager()
    if manager.is_ready and manager.vector_store.retriever is not None:
        return manager.vector_store.retriever
    return None


def hybrid_search(query: str, k: int = 5, threshold: float = 0.3):
    """
    Hybrid search - convenience function.
    Uses a shared RetrieverManager and does NOT reload the DB on each call.
    """
    manager = _get_retriever_manager(score_threshold=threshold)
    if not manager.is_ready:
        return []
    return manager.hybrid_search(query, k, threshold)