# io.py
import pandas as pd
import os
from typing import Optional

def read_papers(filepath: str) -> pd.DataFrame:
    """
    Read papers data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with papers data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Papers file not found: {filepath}")
    
    return pd.read_csv(filepath)

def read_associations(filepath: str) -> pd.DataFrame:
    """
    Read association analysis data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with association analysis data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Associations file not found: {filepath}")
    
    return pd.read_csv(filepath)

def read_risk_analysis(filepath: str) -> pd.DataFrame:
    """
    Read risk analysis data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with risk analysis data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Risk analysis file not found: {filepath}")
    
    return pd.read_csv(filepath)

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save DataFrame to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Path to save CSV file
    index : bool, default=False
        Whether to save row indices
    """
    df.to_csv(filepath, index=index)
    print(f"Data saved to {filepath}")

def load_model_config(config_file: Optional[str] = None) -> dict:
    """
    Load model configuration from file or environment variables.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file
    
    Returns
    -------
    dict
        Dictionary with configuration settings
    """
    config = {
        'model_name': os.getenv('MODEL_NAME', ''),
        'api_base': os.getenv('API_BASE', ''),
        'api_key': os.getenv('API_KEY', ''),
        'log_dir': os.getenv('LOG_DIR', 'logs'),
        'old_openai_api': os.getenv('OLD_OPENAI_API', 'No')
    }
    
    # If config file provided and exists, override with file settings
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config
