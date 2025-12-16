import os
from typing import List, Optional, Tuple, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.tool.rag.embedding import get_embedding_model

class VectorStore:
    """Manages FAISS vector database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.vectorstore = None
        self.retriever = None
    
    def build(self, documents: List[Document]) -> bool:
        """Build vector database from documents"""
        try:
            print('🔄 Creating vector store...')
            embeddings = get_embedding_model()
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            
            os.makedirs(self.db_path, exist_ok=True)
            self.vectorstore.save_local(self.db_path)
            print(f'✅ Vector database saved to {self.db_path}')
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return False
    
    def load(self) -> bool:
        """Load existing vector database"""
        try:
            embeddings = get_embedding_model()
            self.vectorstore = FAISS.load_local(self.db_path, embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 4})
            return True
            
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
            return False
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search using vector similarity"""
        if not self.retriever:
            return []
        
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []
