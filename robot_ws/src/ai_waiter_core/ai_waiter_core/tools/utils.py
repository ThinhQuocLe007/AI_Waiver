from typing import List
from ai_waiter_core.core.utils.logger import logger
from ai_waiter_core.core.schemas import SearchResult

def print_database_summary(documents: List):
    """
    Print a summary of the loaded documents.
    """
    total = len(documents)
    menu_count = sum(1 for d in documents if d.metadata.get('type') == 'menu')
    info_count = sum(1 for d in documents if d.metadata.get('type') == 'info')
    
    logger.info("--- Database Summary ---")
    logger.info(f"Total documents: {total}")
    logger.info(f"Menu items: {menu_count}")
    logger.info(f"Restaurant info docs: {info_count}")
    logger.info("------------------------")

def print_search_results(results: List[SearchResult]):
    """
    Print formatted search results.
    """
    if not results:
        logger.info("No results found.")
        return
        
    logger.info(f"--- Search Results ({len(results)}) ---")
    for i, res in enumerate(results):
        logger.info(f"{i+1}. [{res.score:.4f}] {res.document.page_content[:100]}...")
        logger.info(f"   Source: {res.source} | Type: {res.doc_type}")
    logger.info("----------------------------")
