# rag.py
"""
Retrieval-Augmented Generation (RAG) Module
Responsible for retrieving relevant literature from PubMed and providing it to LLM for reasoning
"""

import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from Bio import Entrez, Medline

class RAGSystem:
    """
    Retrieval-Augmented Generation System
    Responsible for literature retrieval and knowledge base construction
    """
    
    def __init__(self, email: str = 'your_email@example.com', 
                 retry: int = 3, delay: int = 3):
        """
        Initialize RAG system
        
        Parameters
        ----------
        email : str
            Email address for PubMed API
        retry : int
            Number of retry attempts
        delay : int
            Delay between requests in seconds
        """
        self.email = email
        self.retry = retry
        self.delay = delay
        self.logger = logging.getLogger('comed.rag')
        
        # Set Entrez email
        Entrez.email = email
    
    def search_pubmed(self, drug1: str, drug2: str, retmax: int = 30, 
                     sort: str = "relevance") -> pd.DataFrame:
        """
        Search PubMed for literature related to two drugs
        
        Parameters
        ----------
        drug1 : str
            Name of the first drug
        drug2 : str
            Name of the second drug
        retmax : int
            Maximum number of records to retrieve
        sort : str
            Sorting method for results
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing search results with columns:
            'PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'
        """
        query = f'{drug1}[Title/Abstract] AND {drug2}[Title/Abstract]'
        self.logger.info(f"RAG search query: {query}")
        
        # Use ESearch to retrieve document IDs with error retry mechanism
        for attempt in range(1, self.retry + 1):
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort=sort)
                record = Entrez.read(handle)
                handle.close()
                self.logger.info(f"RAG ESearch successful (attempt {attempt})")
                break
            except Exception as e:
                self.logger.error(f"RAG ESearch error (attempt {attempt}): {e}")
                if attempt == self.retry:
                    self.logger.error("RAG ESearch reached maximum retry attempts")
                    return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
                time.sleep(self.delay)
        
        id_list = record.get("IdList", [])
        if not id_list:
            self.logger.info(f"RAG found no related literature: {query}")
            return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
        
        time.sleep(self.delay)
        
        # Use EFetch to get literature information with error retry mechanism
        for attempt in range(1, self.retry + 1):
            try:
                id_str = ",".join(id_list)
                handle = Entrez.efetch(db="pubmed", id=id_str, rettype="medline", retmode="text")
                records = list(Medline.parse(handle))
                handle.close()
                self.logger.info(f"RAG EFetch successful (attempt {attempt})")
                break
            except Exception as e:
                self.logger.error(f"RAG EFetch error (attempt {attempt}): {e}")
                if attempt == self.retry:
                    self.logger.error("RAG EFetch reached maximum retry attempts")
                    return pd.DataFrame(columns=['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 'Publication Date', 'Link'])
                time.sleep(self.delay)
        
        results = []
        for rec in records:
            abstract = rec.get("AB", "").strip()
            if not abstract:
                continue
            
            pmid = rec.get("PMID", "").strip()
            title = rec.get("TI", "").strip()
            authors = rec.get("AU", [])
            
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
        
        self.logger.info(f"RAG retrieved {len(results)} documents")
        return pd.DataFrame(results)
    
    def search_drug_combinations(self, drug_combinations: List[List[str]], 
                               retmax: int = 30, verbose: bool = True) -> pd.DataFrame:
        """
        Search literature for multiple drug combinations
        
        Parameters
        ----------
        drug_combinations : List[List[str]]
            List of drug combination pairs
        retmax : int
            Maximum number of records to retrieve per combination
        verbose : bool
            Whether to display progress bar
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame containing all search results
        """
        all_results = pd.DataFrame()
        
        pbar = tqdm(range(len(drug_combinations)), disable=not verbose,
                   desc="🔍 RAG retrieving literature", unit="combination")
        
        for i in pbar:
            drug1, drug2 = drug_combinations[i]
            pbar.set_description(f"🔍 RAG retrieving: {drug1} + {drug2}")
            
            results = self.search_pubmed(drug1, drug2, retmax=retmax)
            
            if not results.empty:
                results["Drug1"] = drug1
                results["Drug2"] = drug2
                results["Combination_ID"] = i
                all_results = pd.concat([all_results, results], ignore_index=True)
            
            pbar.write(f"  ↳ RAG found {len(results)} papers for {drug1} + {drug2}")
        
        return all_results
    
    def filter_relevant_papers(self, papers: pd.DataFrame, 
                             relevance_threshold: float = 0.5) -> pd.DataFrame:
        """
        Filter relevant papers (based on simple keyword matching)
        
        Parameters
        ----------
        papers : pd.DataFrame
            Papers DataFrame
        relevance_threshold : float
            Relevance threshold
            
        Returns
        -------
        pd.DataFrame
            Filtered papers
        """
        # Here we can implement more complex relevance filtering logic
        # Currently returns all papers
        return papers
    
    def get_retrieval_stats(self, papers: pd.DataFrame) -> Dict[str, Any]:
        """
        Get retrieval statistics
        
        Parameters
        ----------
        papers : pd.DataFrame
            Papers DataFrame
            
        Returns
        -------
        Dict[str, Any]
            Statistics information
        """
        if papers.empty:
            return {
                "total_papers": 0,
                "unique_combinations": 0,
                "papers_per_combination": {}
            }
        
        stats = {
            "total_papers": len(papers),
            "unique_combinations": len(papers[['Drug1', 'Drug2']].drop_duplicates()),
            "papers_per_combination": papers.groupby(['Drug1', 'Drug2']).size().to_dict()
        }
        
        return stats