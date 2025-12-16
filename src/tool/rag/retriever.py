import os
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
    source: str
    doc_type: str


class RetrieverManager:
    """Main class that orchestrates all retrieval operations"""
    
    def __init__(self, config):
        self.config = config 
        self.data_loader = DocumentLoader(self.config)
        
        db_path = self.config.faiss_path
        bm25_path = self.config.bm25_path 
        
        self.vector_store = VectorStore(db_path)
        self.bm25_index = BM25Index(bm25_path, self.data_loader)
        
        self.documents = []
        self.is_ready = False
    
    def build_database(self) -> bool:
        """Build complete database (vector + BM25) from data sources"""
        print('🔄 Building Complete Database (Vector + BM25)')
        
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
        print('🔄 Loading existing database...')
        
        vector_loaded = self.vector_store.load()
        bm25_loaded = self.bm25_index.load()
        
        if vector_loaded and bm25_loaded:
            self.documents = self.bm25_index.documents
            self.is_ready = True
            print('✅ Database loaded successfully')
            return True
        else:
            print('❌ Failed to load database')
            return False
    
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform hybrid search using both BM25 and vector search"""
        if not self.is_ready:
            print("⚠️ Retriever not initialized. Call initialize() first.")
            return []
        
        print(f"🔍 Hybrid searching: '{query}'")
        
        # BM25 search
        bm25_results = self.bm25_index.search(query, k)
        
        # Vector search
        vector_results = self.vector_store.search(query, k)
        
        # Combine results
        combined = self._combine_results(bm25_results, vector_results, k)
        
        print(f"📊 Found {len(combined)} results")
        return combined
    
    def search_by_category(self, query: str, category: str, k: int = 3) -> List[Document]:
        """Search within specific category"""
        all_results = self.hybrid_search(query, k * 2)
        
        # Filter by category
        filtered = [
            doc for doc in all_results 
            if doc.metadata.get('source') == category or doc.metadata.get('type') == category
        ]
        
        return filtered[:k] if filtered else all_results[:k]
    
    def _combine_results(self, bm25_results: List[Tuple[Document, float]], 
                        vector_results: List[Document], k: int) -> List[Document]:
        """Combine BM25 and vector search results"""
        combined = []
        seen_content = set()
        
        # Add BM25 results first (better for keyword matching)
        for doc, score in bm25_results:
            if len(combined) >= k:
                break
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                combined.append(doc)
                seen_content.add(content_hash)
        
        # Fill remaining slots with vector results
        for doc in vector_results:
            if len(combined) >= k:
                break
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                combined.append(doc)
                seen_content.add(content_hash)
        
        return combined
    
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
                        """)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "ready": self.is_ready,
            "total_documents": len(self.documents),
            "menu_items": len([d for d in self.documents if d.metadata.get('source') == 'menu']),
            "restaurant_info": len([d for d in self.documents if d.metadata.get('source') == 'restaurant_info']),
            "vector_store_ready": self.vector_store.vectorstore is not None,
            "bm25_ready": self.bm25_index.bm25 is not None
        }

# Convenience functions 
config = ConfigLoader() 

def build_vector_db():
    """Build vector database - convenience function"""
    retriever_manager = RetrieverManager(config)
    return retriever_manager.build_database()

def get_retriever():
    """Get retriever - convenience function"""
    retriever_manager = RetrieverManager()
    if retriever_manager.load_database():
        return retriever_manager.vector_store.retriever
    return None

def hybrid_search(query: str, k: int = 5):
    """Hybrid search - convenience function"""
    retriever_manager = RetrieverManager(config)
    if retriever_manager.load_database():
        return retriever_manager.hybrid_search(query, k)
    return []

