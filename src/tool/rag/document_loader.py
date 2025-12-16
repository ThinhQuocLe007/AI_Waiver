import json 
import re
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document

class DocumentLoader:
    """Handles loading data from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.menu_path = self.cfg.menu_path
        self.restaurant_info_path = self.cfg.info_path 
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 tokenization - optimized for Vietnamese"""
        text = text.lower()
        # Keep Vietnamese characters, remove special chars but keep spaces
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
        tokens = [token.strip() for token in text.split() if token.strip()]
        return tokens
    
    def load_menu_data(self) -> List[Document]:
        """Load and convert menu items to documents"""
        try:
            with open(self.menu_path, 'r', encoding='utf-8') as file: 
                menu_data = json.load(file)
            
            docs = []
            for item in menu_data:
                content_parts = [
                    f"Món ăn: {item['name']}",
                    f"Giá: {item['price']} VNĐ",
                    f"Mô tả: {item['description']}",
                ]
                
                # Add optional fields if they exist
                if item.get('category'):
                    content_parts.append(f"Phân loại: {item['category']}")
                if item.get('diet_type'):
                    content_parts.append(f"Loại ăn: {item['diet_type']}")
                if item.get('ingredients'):
                    content_parts.append(f"Nguyên liệu: {item['ingredients']}")
                if item.get('taste_profile'):
                    content_parts.append(f"Hương vị: {item['taste_profile']}")
                if item.get('tags'):
                    content_parts.append(f"Đặc điểm: {item['tags']}")
                
                content = '\n'.join(content_parts)
                
                # Create search keywords
                search_keywords = f"{item['name']} {item.get('category', '')} {item.get('tags', '')}".lower()
                
                docs.append(Document(
                    page_content=content.strip(), 
                    metadata={
                        **item, 
                        "type": "menu_item", 
                        "source": "menu",
                        "search_keywords": search_keywords
                    }
                ))
            
            print(f"✅ Loaded {len(menu_data)} menu items")
            return docs
            
        except Exception as e:
            print(f"❌ Error loading menu: {e}")
            return []
    
    def load_restaurant_info(self) -> List[Document]:
        """Load restaurant information from text file"""
        try:
            with open(self.restaurant_info_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            docs = []
            
            # Split content by sections (## headers)
            sections = re.split(r'^## \d+\.\s*(.+?)$', content, flags=re.MULTILINE)
            
            # Process sections
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    section_title = sections[i].strip()
                    section_content = sections[i + 1].strip()
                    
                    if section_content:
                        full_content = f"{section_title}\n{section_content}"
                        
                        # Determine section type
                        section_type = self._classify_section_type(section_title)
                        
                        docs.append(Document(
                            page_content=full_content,
                            metadata={
                                "type": section_type,
                                "source": "restaurant_info",
                                "section_title": section_title,
                                "search_keywords": f"{section_title} {section_content}".lower()
                            }
                        ))
            
            # Add quick-access documents for common queries
            print(f"✅ Loaded {len(docs)} restaurant info sections")
            return docs
            
        except Exception as e:
            print(f"❌ Error loading restaurant info: {e}")
            return []
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ['giờ', 'thời gian', 'hoạt động']):
            return "hours"
        elif any(keyword in title_lower for keyword in ['địa chỉ', 'liên hệ', 'contact']):
            return "location"
        elif any(keyword in title_lower for keyword in ['tiện ích', 'dịch vụ', 'amenities']):
            return "services"
        elif any(keyword in title_lower for keyword in ['chính sách', 'policy']):
            return "policies"
        else:
            return "general_info"