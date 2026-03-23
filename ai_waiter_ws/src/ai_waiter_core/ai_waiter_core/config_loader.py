import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config = self._load_yaml(config_path)
        
    def _load_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @property
    def menu_path(self):
        return Path(self.config['paths']['input_data']['menu_file'])
    
    @property
    def info_path(self): 
        return Path(self.config['paths']['input_data']['restaurant_info'])
    
    @property
    def faiss_path(self):
        return Path(self.config['paths']['vector_store']['faiss_db'])
    
    @property
    def bm25_path(self): 
        return Path(self.config['paths']['vector_store']['bm25_file'])
    
    @property
    def device(self):
        return self.config['system']['device']
