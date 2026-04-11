import json 
import os 
from langchain_core.documents import Document
from ai_waiter_core.core.config import settings
from ai_waiter_core.core.utils.logger import logger



class DocumentLoader: 
    def __init__(self): 
        self.parsers = {
            "menu.json": self._parse_menu_json,
            "restaurant_info.txt": self._parse_info_text,
        }


    def load(self, file_path): 

        # Select the parser based on the filename
        filename = os.path.basename(file_path)
        parser = self.parsers.get(filename)

        if not parser: 
            logger.warning(f"No parser found for {filename}, using default loader")
            return self._default_text_loader(file_path)

        # Call the parser
        try:
            return parser(file_path)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return [] # return None if error 
        
        

    def _parse_menu_json(self, file_path): 
        """
        Parse the menu JSON file and return a list of Document objects.

        Args:
            file_path (str): The path to the menu JSON file.

        Returns:
            List[Document]: A list of Document objects.
        """
        with open(file_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)

        documents = []
        for item in data: 
            # Create a metadata dictionary
            metadata = {
                "source": "menu.json",
                "name": item.get("name"),
                "price": item.get("price"),
                "diet_type": item.get("diet_type"),
                "category": item.get("category"),
            }

            # Create page_content 
            page_content = f"""
            Tên món: {item.get('name')}
            Mô tả: {item.get('description')}
            Giá: {item.get('price')}
            Loại món ăn: {item.get('diet_type')}
            Danh mục: {item.get('category')}
            Thành phần: {item.get('ingredients')}
            Hương vị: {item.get('taste_profile')}
            Tags: {item.get('tags')}
            """

            # Create a Document object 
            docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        return docs

    def _parse_info_text(self, file_path): 
        """
        Parse the restaurant info text file and return a list of Document objects.
        The data format should like: 
        ## Title
        Content
        ## Title
        Content
        
        Args:
            file_path (str): The path to the restaurant info text file.

        Returns:
            List[Document]: A list of Document objects.
        """
        with open(file_path, 'r', encoding='utf-8') as f: 
            content = f.read()

        # Split content by ## section
        sections = content.split('##')
        docs = [] 
        for section in sections: 
            if not section.strip(): 
                continue

            # Split section into title and content
            parts = section.split('\n', 1)
            if len(parts) < 2: 
                continue

            title = parts[0].strip()
            content = parts[1].strip()

            # Create metadata
            metadata = {
                "source": "restaurant_info.txt",
                "title": title,
            }

            # Create a Document object
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return documents

    def _default_text_loader(self, file_path):
        """
        Load a text file and return a list of Document objects.
        
        Args:
            file_path (str): The path to the text file.

        Returns:
            List[Document]: A list of Document objects.
        """
        with open(file_path, 'r', encoding='utf-8') as f: 
            content = f.read()

        # Split content by lines
        lines = content.split('\n')
        docs = [] 
        for line in lines: 
            if not line.strip(): 
                continue

            # Create metadata
            metadata = {
                "source": os.path.basename(file_path),
            }

            # Create a Document object
            docs.append(
                Document(
                    page_content=line,
                    metadata=metadata,
                )
            )
        return docs

    