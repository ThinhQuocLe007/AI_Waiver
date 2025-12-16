import os
import pickle
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.tool.rag.document_loader import DocumentLoader

from config.config_loader import ConfigLoader 

class BM25Index:
    """Manages BM25 index operations"""
    def __init__(self,bm25_path, data_loader):
        self.index_path = bm25_path 
        self.data_loader = data_loader
        self.bm25 = None
        self.documents = []
        self.corpus = []
    
    def build(self, documents: List[Document]) -> bool:
        """Build BM25 index from documents"""
        try:
            print('🔄 Building BM25 index...')
            
            self.documents = documents
            self.corpus = []
            
            for doc in documents:
                # Create searchable text
                searchable_text = doc.page_content
                
                # Add metadata keywords
                if doc.metadata.get('name'):
                    searchable_text += f" {doc.metadata['name']}"
                if doc.metadata.get('search_keywords'):
                    searchable_text += f" {doc.metadata['search_keywords']}"
                
                tokens = self.data_loader.preprocess_text(searchable_text)
                self.corpus.append(tokens)
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.corpus)
            
            # Save index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'corpus': self.corpus
                }, f)
            
            print(f'✅ BM25 index saved with {len(documents)} documents')
            return True
            
        except Exception as e:
            print(f"❌ Error creating BM25 index: {e}")
            return False
    
    def load(self) -> bool:
        """Load existing BM25 index"""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.corpus = data['corpus']
            return True
            
        except FileNotFoundError:
            print(f"❌ BM25 index not found at {self.index_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading BM25 index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search using BM25"""
        if not self.bm25:
            return []
        
        try:
            query_tokens = self.data_loader.preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top results with scores
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            results = []
            
            for idx in top_indices[:k * 2]:  # Get more candidates
                if scores[idx] > 0:
                    results.append((self.documents[idx], scores[idx]))
            
            return results[:k]
            
        except Exception as e:
            print(f"❌ BM25 search error: {e}")
            return []