import os
from typing import List, Tuple
from ai_waiter_core.core.config import settings
from ai_waiter_core.core.utils.logger import logger
from ai_waiter_core.core.schemas.search import SearchResult
from langchain_core.documents import Document

from .engines.bm25 import BM25Index
from .engines.vector_db import VectorStore
from .engines.document_loader import DocumentLoader
from .engines.scoring import normalize_vector_score
from .engines.fusion import get_fusion_strategy


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
        
        # print_database_summary(self._documents)

        # Build search engines with the combined document list
        if self.vector_engine.build(self._documents) and self.bm25_engine.build(self._documents):
            self.is_ready = True 
            logger.info("[INFO] Database built successfully")
            return True 

        return False 

    # Hybrid search
    def hybrid_search(self, query: str, k: int = None, threshold: float = None, mode: str = "rrf", rrf_k: int = 60) -> List[SearchResult]:
        """
        Hybrid search using BM25 and vector search
        Args:
            query: Query string
            k: Number of results to return
            threshold: Score threshold
            mode: Search algorithm mode ('rrf' or 'weighted')
            rrf_k: Constant for RRF logic (default 60)
        Returns:
            List of SearchResult objects
        """
        if not self.is_ready:
            logger.warning("Retriever not ready. Run build or load first.")
            return []
        
        k = k or self.k
        threshold = threshold or self.score_threshold
        
        # 1. Get raw scores from individual engines
        bm25_raw, vector_raw = self._get_raw_scores(query, k=k*2)
        
        # 2. Delegate fusion to the strategy module
        strategy = get_fusion_strategy(mode)
        return strategy.fuse(
            bm25_results=bm25_raw, 
            vector_results=vector_raw, 
            k=k, 
            threshold=threshold,
            rrf_k=rrf_k
        )

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

    # --- 3. Utilities ---

    def load_database(self) -> bool: 
        logger.info("[INFO] Loading database from disk...")
        if self.vector_engine.load() and self.bm25_engine.load():
            self.is_ready = True 
            return True 
        return False
