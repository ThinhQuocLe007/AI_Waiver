from sentence_transformers import SentenceTransformer, util
import faiss 
import numpy as np
import torch
import json
import os 

class RAGSystem:
    def __init__(self, menu_file_path, model_name='all-MiniLM-L6-v2'): 
        """
        Initialize the RAG system 
        """
        self.menu_file_path = menu_file_path
        self.model_name = model_name
        self.menu_data = [] 

        self.encoder = None # SentenceTransformer model
        self.index = None # FAISS index
        self.embeddings = None # Store embeddings

        try: 
            self.menu_data = self._load_menu(menu_file_path)
            print(f'Loaded {len(self.menu_data)} menu items')

            # Load embedding model
            print(f'Loading embedding model {self.model_name}...')
            self.encoder = SentenceTransformer(self.model_name) 

            self.index, self.embeddings = self._build_faiss_index()
            print('✅ RAG system initialized successfully.')

        except Exception as e:
            print(f'❌ Error initializing RAG system: {e}')
            raise e
            
    def _load_menu(self, file_path):
        """
        Load menu from file ( support text and json formats)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Menu file not found: {file_path}')
        
        try: 
            with open(file_path, 'r', encoding= 'utf-8') as f:
                content = f.read() 

            if file_path.endswith('.json'):
                # Fixed: Return JSON data directly
                return json.loads(content)
            
            # Parse text format 
            items_raw = content.strip().split('---')
            menu = [] 
            for item_raw in items_raw: 
                if not item_raw.strip():
                    continue

                item_dict = {}
                for line in item_raw.strip().split('\n'):
                    if ':' not in line: 
                        continue

                    key, value = line.split(':', 1)
                    item_dict[key.strip().lower()] = value.strip()
                
                if self._validate_menu_item(item_dict):
                    menu.append(item_dict)

            return menu 

        except Exception as e:
            print(f'Error loading menu: {e}')
            raise e

    def _validate_menu_item(self, item):
        """Validate that menu item has required fields"""
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in item or not item[field]:
                print(f"Warning: Menu item missing {field}: {item}")
                return False
        return True

    def _build_faiss_index(self):
        """
        Build faiss index from menu descriptions
        """
        if not self.menu_data: 
            raise ValueError('No menu data available to build index')
        
        # Fixed: Variable name from description to descriptions
        descriptions = [] 
        for item in self.menu_data: 
            # Fixed: Syntax error in f-string
            desc = f"{item['name']}: {item['description']}"
            if 'category' in item: 
                desc += f" Category: {item['category']}"
            if 'ingredients' in item: 
                # Fixed: Typo from Ingredident to Ingredients
                desc += f" Ingredients: {item['ingredients']}"
            descriptions.append(desc)
            
        print('Building search index for menu items')

        # Fixed: Variable name from description to descriptions
        embeddings = self.encoder.encode(descriptions, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Create index 
        index = faiss.IndexFlatIP(embeddings.shape[1])

        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        print(f"✅ Index built with {index.ntotal} items")    
        return index, embeddings
    
    def search_index(self, query, top_k=5, threshold=0.3):
        """
        Search for the relevant menu items 

        Args: 
            query (str): Search query 
            top_k (int): number of results to return 
            threshold (float): minimum similarity threshold (cosine similarity)
        Return: 
            list: list of relevant menu items with scores
        """
        if not self.index or not self.encoder: 
            raise ValueError('RAG system not properly initialized')

        try: 
            query_embedding = self.encoder.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search 
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.menu_data)))

            # Filter by threshold and format the result 
            results = [] 
            for score, idx in zip(scores[0], indices[0]): 
                if score >= threshold:  # Fixed: >= instead of >
                    item = self.menu_data[idx]
                    result = {
                        'name': item['name'], 
                        'description': item['description'], 
                        # Fixed: Use .get() to handle missing fields
                        'price': item.get('price', 'N/A'), 
                        'category': item.get('category', 'N/A'), 
                        'similarity_score': float(score)
                    }
                    results.append(result)
            
            return results 
        except Exception as e: 
            print(f"[ERROR] Happened during search: {e}")
            return []  # Return [] for graceful degradation

    def get_context_for_llms(self, query, top_k=5):
        """
        Get formatted context for LLMs 
        """
        # Fixed: Method name from search to search_index
        results = self.search_index(query, top_k)

        if not results: 
            return 'No relevant menu items found'
        
        context = "Here are the relevant menu items:\n\n"
        for i, item in enumerate(results, 1):
            context += f"{i}. **{item['name']}** (Price: {item['price']} VND)\n"
            context += f"   Description: {item['description']}\n"
            if item['category'] != 'N/A':
                context += f"   Category: {item['category']}\n"
            context += f"   Relevance: {item['similarity_score']:.2f}\n\n"
        
        return context  

    def add_menu_item(self, item):
        """Add a new menu item and rebuild index"""
        if self._validate_menu_item(item):
            self.menu_data.append(item)
            self.index, self.embeddings = self._build_faiss_index()
            print(f"Added item: {item['name']}")
            return True
        else:
            print("Invalid menu item format")
            return False

    def save_menu_data(self, file_path):
        """Save current menu data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.menu_data, f, ensure_ascii=False, indent=2)
            print(f"Menu data saved to {file_path}")
        except Exception as e:
            print(f"Error saving menu data: {e}")

    def get_stats(self):
        """Get system statistics"""
        return {
            'total_items': len(self.menu_data),
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_size': self.index.ntotal if self.index else 0
        }