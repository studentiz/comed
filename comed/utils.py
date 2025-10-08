# utils.py
import re
import json
import difflib
import logging
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace while preserving structure.
    
    Parameters
    ----------
    text : str
        Text to clean
    
    Returns
    -------
    str
        Cleaned text
    """
    # Remove leading/trailing whitespace from each line, preserving newlines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON data from text.
    
    Parameters
    ----------
    text : str
        Text potentially containing JSON
    
    Returns
    -------
    dict or None
        Extracted JSON as dictionary, or None if not found/invalid
    """
    # Look for text that appears to be JSON (between curly braces)
    json_pattern = r'(\{.*?\})'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        return None
    
    # Try each potential JSON match
    for match in matches:
        try:
            json_obj = json.loads(match)
            return json_obj
        except json.JSONDecodeError:
            continue
    
    return None

def remove_code_markers(text: str) -> str:
    """
    Remove code block markers from text.
    
    Parameters
    ----------
    text : str
        Text with potential code markers
    
    Returns
    -------
    str
        Cleaned text without code markers
    """
    # Remove triple backticks and language specifiers
    text = re.sub(r'```(?:json|python|bash|)\n', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def find_similar_items(items: List[str], query: str, threshold: float = 0.8) -> List[str]:
    """
    Find items similar to query string.
    
    Parameters
    ----------
    items : List[str]
        List of items to search
    query : str
        Query string to match
    threshold : float, default=0.8
        Similarity threshold (0-1)
    
    Returns
    -------
    List[str]
        List of similar items
    """
    similar_items = []
    for item in items:
        similarity = difflib.SequenceMatcher(None, item.lower(), query.lower()).ratio()
        if similarity >= threshold:
            similar_items.append(item)
    return similar_items

def configure_logging(log_file: Optional[str] = None, 
                     log_level: int = logging.INFO) -> None:
    """
    Configure logging for CoMed.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
    log_level : int, default=logging.INFO
        Logging level
    """
    # Create logger
    logger = logging.getLogger('comed')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def get_unique_drug_combinations(df: pd.DataFrame) -> List[List[str]]:
    """
    Get unique drug combinations from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Drug1 and Drug2 columns
    
    Returns
    -------
    List[List[str]]
        List of unique drug combinations
    """
    if 'Drug1' not in df.columns or 'Drug2' not in df.columns:
        raise ValueError("DataFrame must contain 'Drug1' and 'Drug2' columns")
    
    # Get unique combinations
    unique_combinations = df[['Drug1', 'Drug2']].drop_duplicates().values.tolist()
    return unique_combinations
