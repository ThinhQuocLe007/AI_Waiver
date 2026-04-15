import os 
import pickle 
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ai_waiter_core.core.utils.logger import logger
from .base import SearchIndex

class BM25Index(SearchIndex):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
    
    def build(self, documents: List[Document]) -> bool:
        """
        Build BM25 index from documents
        Args:
            documents (List[Document]): List of documents to build index from
        Returns:
            bool: True if index was built successfully, False otherwise
        """
        try:
            self.documents = documents
            self.tokenized_docs = []
            for doc in documents:
                # Title-Only BM25 search strategy for precise dish name lookups
                title = doc.metadata.get("name") or doc.metadata.get("title")
                
                if title:
                    self.tokenized_docs.append(str(title).lower().split())
                else:
                    self.tokenized_docs.append(doc.page_content.lower().split())
                    
            self.bm25 = BM25Okapi(self.tokenized_docs)
            self.save()
            logger.info(f'[INFO] BM25 index built and saved to {self.db_path}')
            return True

        except Exception as e:
            logger.error(f'[ERROR] Creating BM25 index: {e}')
            return False
    
    def load(self) -> bool:
        """
        Load BM25 index from disk
        """
        try:
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
                self.tokenized_docs = data['tokenized_docs']
            logger.info(f'[INFO] BM25 index loaded from {self.db_path}')
            return True
        except Exception as e:
            logger.error(f'[ERROR] Loading BM25 index: {e}')
            return False  
    
    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Search BM25 index for query
        Args:
            query (str): Query string
            k (int): Number of results to return
        Returns:
            List[Tuple[Document, float]]: List of documents and their scores
        """
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)


            # Get top-k
            doc_scores = []
            for idx, score in enumerate(scores):
                doc_scores.append((self.documents[idx], float(score)))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Return 
            return doc_scores[:k]
        except Exception as e:
            logger.error(f'[ERROR] Searching BM25 index: {e}')
            return []

    def save(self) -> bool:
        """
        Save BM25 index to disk
        """
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f'[INFO] BM25 index saved to {self.db_path}')
            return True
        except Exception as e:
            logger.error(f'[ERROR] Saving BM25 index: {e}')
            return False