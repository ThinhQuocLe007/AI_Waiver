import os
from typing import List, Tuple
from ai_waiter_core.core.config import settings
from ai_waiter_core.core.utils.logger import logger
from ai_waiter_core.core.schemas.search import SearchResult
from langchain_core.documents import Document

from .search_engines.bm25 import BM25Index
from .search_engines.vector_db import VectorStore
from .data.document_loader import DocumentLoader
from .search_engines.scoring import (
    calculate_hybrid_score, 
    normalize_vector_score, 
    normalize_bm25_batch
)
from .utils import print_database_summary

class RetrieverManager: 
    def __init__(self, score_threshold: float = 0.3, k: int = 5): 
        self.loader = DocumentLoader()
        self.score_threshold = score_threshold
        self.k = k

        # Initialize engines 
        self.vector_engine = VectorStore(db_path=settings.VECTOR_DB_PATH)
        self.bm25_engine = BM25Index(db_path=settings.BM25_PATH)

        self.is_ready = False 
        self._documents= [] 

    # Load directory
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Scans a directory and loads every file that we have a parser for.
        """
        all_documents = []
        
        # Loop through all the files in folder 
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Process the files onnly 
            if os.path.isfile(file_path):
                # Use the document loader to parser the file
                docs = self.loader.load(file_path)
                all_documents.extend(docs)
                
        logger.info(f"Scaled Loading: Found {len(all_documents)} docs in {directory_path}")
        return all_documents

    # Build database
    def build_database(self, data_paths: List[str]) -> bool: 
        """
        Build the database from a list of files or directories 
        Args:
            data_paths: List of file or directory paths to build the database from
        Returns:
            True if the database was built successfully, False otherwise
        """
        logger.info(f"[INFO] Building database from {len(data_paths)} paths...")
        self._documents = []

        for path in data_paths:
            if os.path.isdir(path):
                self._documents.extend(self.load_directory(path))
            elif os.path.isfile(path):
                self._documents.extend(self.loader.load(path))
        
        if not self._documents: 
            logger.error("[ERROR] No documents successfully loaded")
            return False 
        
        print_database_summary(self._documents)

        # Build search engines with the combined document list
        if self.vector_engine.build(self._documents) and self.bm25_engine.build(self._documents):
            self.is_ready = True 
            logger.info("[INFO] Database built successfully")
            return True 

        return False 

    # Hybrid search
    def hybrid_search(self, query: str, k: int = None, threshold: float = None) -> List[SearchResult]:
        """
        Hybrid search using BM25 and vector search
        Args:
            query: Query string
            k: Number of results to return
            threshold: Score threshold
        Returns:
            List of SearchResult objects
        """
        if not self.is_ready:
            logger.warning("Retriever not ready. Run build or load first.")
            return []
        
        k = k or self.k
        threshold = threshold or self.score_threshold
        
        # Get raw scores
        bm25_raw, vector_raw = self._get_raw_scores(query, k=k*2)
        
        # Merge based on content hash
        combined_map = self._merge_scores(bm25_raw, vector_raw)
        
        # Apply math and format result objects
        return self._rank_and_format(combined_map, threshold)[:k]

    def _get_raw_scores(self, query: str, k: int) -> Tuple[list, list]:
        """
        Internal helper to call engines and normalize vector distances
        Args:
            query: Query string
            k: Number of results to return
        Returns:
            Tuple of (bm25_results, vector_results)
        """
        bm25 = self.bm25_engine.search(query, k=k)
        vector = self.vector_engine.search(query, k=k)
        
        # Convert FAISS distance to 0-1 similarity score
        vector_norm = [(doc, normalize_vector_score(s)) for doc, s in vector]
        return bm25, vector_norm

    def _merge_scores(self, bm25_results, vector_results) -> dict:
        """
        Merge results from both engines into a single lookup map
        Args:
            bm25_results: BM25 results
            vector_results: Vector results
        Returns:
            Dictionary of merged results: {doc_id: {'doc': doc, 'bm25': score, 'vector': score}}
        """
        merged = {}
        
        # Add BM25 findings
        for doc, score in bm25_results:
            doc_id = hash(doc.page_content)
            merged[doc_id] = {'doc': doc, 'bm25': score, 'vector': 0.0}
            
        # Merge with Vector findings
        for doc, score in vector_results:
            doc_id = hash(doc.page_content)
            if doc_id in merged:
                merged[doc_id]['vector'] = score
            else:
                merged[doc_id] = {'doc': doc, 'bm25': 0.0, 'vector': score}
        return merged

    def _rank_and_format(self, combined_map: dict, threshold: float) -> List[SearchResult]: #TODO: DEO HIEU
        """
        Orchestrates the final scoring and conversion to SearchResult objects
        Args:
            combined_map: Dictionary of merged results
            threshold: Score threshold
        Returns:
            List of SearchResult objects
        """
        # Extract raw BM25 scores to normalize them as a batch
        raw_bm25_scores = [v['bm25'] for v in combined_map.values() if v['bm25'] > 0]
        norm_bm25_scores = normalize_bm25_batch(raw_bm25_scores)
        
        # Map back to entries (this creates a temporary map for normalized values)
        norm_map = { 
            val: norm 
            for val, norm in zip([v['bm25'] for v in combined_map.values() if v['bm25'] > 0], norm_bm25_scores)
        }

        final_list = []
        for entry in combined_map.values():
            # Get normalized BM25 score (default 0.0 if not found)
            bm25_n = norm_map.get(entry['bm25'], 0.0)
            
            # Final Hybrid calculation
            score = calculate_hybrid_score(bm25_n, entry['vector'])
            
            if score >= threshold:
                final_list.append(SearchResult(
                    document=entry['doc'],
                    score=score,
                    bm25_score=entry['bm25'],
                    vector_score=entry['vector'],
                    source=entry['doc'].metadata.get('source', 'unknown'),
                    doc_type=entry['doc'].metadata.get('type', 'unknown')
                ))
                
        return sorted(final_list, key=lambda x: x.score, reverse=True)

    # --- 3. Utilities ---

    def load_database(self) -> bool: 
        logger.info("[INFO] Loading database from disk...")
        if self.vector_engine.load() and self.bm25_engine.load():
            self.is_ready = True 
            return True 
        return False
