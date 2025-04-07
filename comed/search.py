# search.py
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from Bio import Entrez, Medline

def search_pubmed(drug1: str, drug2: str, retmax: int = 30, sort: str = "relevance", 
                 email: str = 'your_email@example.com', retry: int = 3, 
                 delay: int = 3) -> pd.DataFrame:
    """
    Search PubMed for literature related to two drugs based on their names.
    
    Parameters
    ----------
    drug1 : str
        Name of the first drug
    drug2 : str
        Name of the second drug
    retmax : int, default=30
        Maximum number of records to retrieve
    sort : str, default="relevance"
        Sorting method for results
    email : str, default='your_email@example.com'
        Email address for Entrez API
    retry : int, default=3
        Maximum number of retry attempts
    delay : int, default=3
        Delay between requests in seconds
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing search results with columns:
        'PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'
    """
    # Set Entrez email
    Entrez.email = email
    
    # Construct query string
    query = f'{drug1}[Title/Abstract] AND {drug2}[Title/Abstract]'
    logging.info(f"Starting search, query: {query}")
    
    # Use ESearch to retrieve document IDs with error retry mechanism
    for attempt in range(1, retry + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort)
            record = Entrez.read(handle)
            handle.close()
            logging.info(f"ESearch request successful (attempt {attempt})")
            break
        except Exception as e:
            logging.error(f"ESearch request error (attempt {attempt}): {e}")
            if attempt == retry:
                logging.error("ESearch reached maximum retry attempts, exiting.")
                return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
            time.sleep(delay)
    
    id_list = record.get("IdList", [])
    if not id_list:
        logging.info(f"No related literature found: {query}")
        return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
    
    # Request pause to avoid continuous requests to PubMed server
    time.sleep(delay)
    
    # Use EFetch to get literature information with error retry mechanism
    for attempt in range(1, retry + 1):
        try:
            id_str = ",".join(id_list)
            handle = Entrez.efetch(db="pubmed", id=id_str, rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            handle.close()
            logging.info(f"EFetch request successful (attempt {attempt})")
            break
        except Exception as e:
            logging.error(f"EFetch request error (attempt {attempt}): {e}")
            if attempt == retry:
                logging.error("EFetch reached maximum retry attempts, exiting.")
                return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
            time.sleep(delay)
    
    results = []
    for rec in records:
        abstract = rec.get("AB", "").strip()
        # Skip if no abstract
        if not abstract:
            continue
        
        pmid = rec.get("PMID", "").strip()
        title = rec.get("TI", "").strip()
        authors = rec.get("AU", [])
        
        # Skip if no authors
        if not authors:
            continue
            
        authors_str = ", ".join(authors) if authors else ""
        journal = rec.get("JT", "").strip()
        pub_date = rec.get("DP", "").strip()
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        
        results.append({
            "PMID": pmid,
            "Title": title,
            "Abstract": abstract,
            "Authors": authors_str,
            "Journal": journal,
            "Publication Date": pub_date,
            "Link": link
        })
    
    logging.info(f"A total of {len(results)} documents with abstracts were obtained")
    
    # Create pandas DataFrame from results
    df = pd.DataFrame(results)
    return df

def search_papers_for_drug_combinations(drugs_combinations: List[List[str]], 
                                      filepath: str = "ddc_pappers.csv", 
                                      verbose: bool = True, 
                                      retmax: int = 30, 
                                      email: str = 'your_email@example.com', 
                                      retry: int = 3, 
                                      delay: int = 3) -> pd.DataFrame:
    """
    Search papers for each drug combination and compile results.
    
    Parameters
    ----------
    drugs_combinations : List[List[str]]
        List of drug combination pairs
    filepath : str, default="ddc_pappers.csv"
        Path to save search results
    verbose : bool, default=True
        Whether to display progress bar
    retmax : int, default=30
        Maximum number of records to retrieve per combination
    email : str, default='your_email@example.com'
        Email address for Entrez API
    retry : int, default=3
        Maximum number of retry attempts
    delay : int, default=3
        Delay between requests in seconds
    
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame containing all search results
    """
    ddc_pappers = pd.DataFrame()
    drugs_count = len(drugs_combinations)
    
    # Create progress bar with descriptive prefix
    pbar = tqdm(range(drugs_count), disable=not verbose, 
                desc="📚 Searching PubMed for papers", 
                unit="combination")
    
    for i in pbar:
        drugs_combination_item = drugs_combinations[i]
        drug1 = drugs_combination_item[0]
        drug2 = drugs_combination_item[1]
        
        # Update progress bar description with current drug combination
        pbar.set_description(f"📚 Searching: {drug1} + {drug2}")
        
        df_results = search_pubmed(
            drug1, drug2, 
            retmax=retmax, 
            email=email, 
            retry=retry, 
            delay=delay
        )
        
        papers_found = df_results.shape[0]
        # Display papers found for this combination
        pbar.write(f"  ↳ Found {papers_found} papers for {drug1} + {drug2}")
        
        df_results["ID"] = i
        df_results["Drug1"] = drug1
        df_results["Drug2"] = drug2
        df_results["Papers_count"] = papers_found
        
        ddc_pappers = pd.concat([ddc_pappers, df_results], ignore_index=True)
        
        # Save incremental results
        ddc_pappers.to_csv(filepath, index=False)
    
    return ddc_pappers
