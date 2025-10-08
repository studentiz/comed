# cot.py
"""
Chain-of-Thought (CoT) Reasoning Module
Responsible for structured reasoning and step-by-step analysis
"""

import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from .analysis import invoke_openai_chat_completion, remove_think_tags_from_text, remove_code_block_tags

class CoTReasoner:
    """
    Chain-of-Thought Reasoner
    Responsible for structured reasoning and step-by-step analysis
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        """
        Initialize CoT reasoner
        
        Parameters
        ----------
        model_name : str
            LLM model name
        api_key : str, optional
            API key
        api_base : str, optional
            API base URL
        old_openai_api : str
            Whether to use old OpenAI API format
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.old_openai_api = old_openai_api
        self.logger = logging.getLogger('comed.cot')
        
        # Define standard reasoning steps
        self.reasoning_steps = [
            "Identify sentences that explicitly mention the combined use of {entity_1} and {entity_2}. If there is no corresponding sentence, please honestly output 'not mentioned'.",
            "What do you think of the association between {entity_1} and {entity_2}? If there is no association, please output 'no association'.",
            "Wait, is the combined use of {entity_1} and {entity_2} being over-interpreted?",
            "Please finally determine whether the abstract mentions the combined use of {entity_1} and {entity_2}, and explain the reasons."
        ]
    
    def analyze_drug_association(self, abstract: str, drug1: str, drug2: str,
                                custom_steps: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Analyze drug association
        
        Parameters
        ----------
        abstract : str
            Literature abstract
        drug1 : str
            First drug name
        drug2 : str
            Second drug name
        custom_steps : List[str], optional
            Custom reasoning steps
            
        Returns
        -------
        Tuple[str, str]
            (result, reasoning_process)
        """
        steps = custom_steps if custom_steps else self.reasoning_steps
        formatted_steps = [step.format(entity_1=drug1, entity_2=drug2) for step in steps]
        
        question = f"Does the text indicate an association between {drug1} and {drug2}?"
        
        # Execute step-by-step reasoning
        reasoning_results, final_result = self._perform_stepwise_reasoning(
            formatted_steps, abstract, question
        )
        
        # Format output
        format_description = f"""Output in JSON format with two fields: "result" and "reason". If the abstract mentions the combination of {drug1} and {drug2}, set "result" to "yes"; otherwise, set it to "no". Provide a detailed explanation in the "reason" field."""
        format_example = '{"result":"yes", "reason":"xxxxx"}'
        
        formatted_result = self._format_output(
            final_result, format_description, format_example
        )
        
        return formatted_result, " ".join(reasoning_results)
    
    def _perform_stepwise_reasoning(self, steps: List[str], context: str, 
                                   question: str) -> Tuple[List[str], str]:
        """
        Perform step-by-step reasoning
        
        Parameters
        ----------
        steps : List[str]
            Reasoning steps
        context : str
            Context information
        question : str
            Question to be answered
            
        Returns
        -------
        Tuple[List[str], str]
            (reasoning results list, final result)
        """
        messages = [
            {"role": "system", "content": (
                "You are an excellent language master. You cannot request additional information from the user or suggest that they look for more information."
            )},
        ]
        
        results = []
        for i, step in enumerate(steps):
            messages.append({"role": "user", "content": step})
            results.append(step + "\n")
            
            step_result = invoke_openai_chat_completion(
                self.model_name, messages,
                api_key=self.api_key,
                api_base=self.api_base,
                old_openai_api=self.old_openai_api
            )
            
            self.logger.info(f"【CoT reasoning step {i+1}】\n{step_result}\n{'-'*50}")
            results.append(f"**Answer{i+1}**\n" + step_result + "\n")
            messages.append({"role": "assistant", "content": step_result})
        
        return results, step_result
    
    def _format_output(self, content: str, format_requirements: str, 
                      format_example: str) -> str:
        """
        Format output
        
        Parameters
        ----------
        content : str
            Content to format
        format_requirements : str
            Format requirements
        format_example : str
            Format example
            
        Returns
        -------
        str
            Formatted output
        """
        input_messages = [
            {"role": "system", "content": (
                "You are a formatting function. You will only output the content in the transformed format and nothing else."
            )},
            {"role": "user", "content": f"**Content**:\n{content}\n**Format Requirements**:{format_requirements}\n**Format Example**:{format_example}"}
        ]
        
        formatted_result = invoke_openai_chat_completion(
            self.model_name, input_messages,
            api_key=self.api_key,
            api_base=self.api_base,
            old_openai_api=self.old_openai_api
        )
        
        if formatted_result is None:
            return None
        
        cleaned_result = remove_code_block_tags(formatted_result)
        
        try:
            json_obj = json.loads(cleaned_result)
            return json.dumps(json_obj, ensure_ascii=False)
        except json.JSONDecodeError as e:
            self.logger.error(f"CoT JSON parsing failed: {e}")
            return cleaned_result
    
    def batch_analyze_associations(self, papers: pd.DataFrame, 
                                  max_retries: int = 30, 
                                  retry_delay: int = 5,
                                  verbose: bool = True) -> pd.DataFrame:
        """
        Batch analyze drug associations
        
        Parameters
        ----------
        papers : pd.DataFrame
            Papers DataFrame
        max_retries : int
            Maximum number of retry attempts
        retry_delay : int
            Retry delay
        verbose : bool
            Whether to display progress bar
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing association analysis results
        """
        results = []
        
        pbar = tqdm(range(len(papers)), disable=not verbose,
                   desc="🧠 CoT analyzing drug associations", unit="paper")
        
        for i in pbar:
            paper = papers.iloc[i]
            drug1 = paper["Drug1"]
            drug2 = paper["Drug2"]
            abstract = paper["Abstract"]
            
            pbar.set_description(f"🧠 CoT analyzing: {drug1} + {drug2} [{i+1}/{len(papers)}]")
            
            # Execute CoT analysis
            result, reasoning = self._analyze_with_retry(
                abstract, drug1, drug2, max_retries, retry_delay
            )
            
            # Parse result
            try:
                result_data = json.loads(result)
                association_result = result_data.get('result', 'error')
                reason = result_data.get('reason', 'error')
            except json.JSONDecodeError:
                association_result = 'error'
                reason = 'JSON parsing error'
            
            # Record result
            result_row = {
                "ID": i,
                "Drug1": drug1,
                "Drug2": drug2,
                "PMID": paper.get("PMID", ""),
                "Title": paper.get("Title", ""),
                "Abstract": abstract,
                "Authors": paper.get("Authors", ""),
                "Journal": paper.get("Journal", ""),
                "Publication Date": paper.get("Publication Date", ""),
                "Link": paper.get("Link", ""),
                "Reasoning": reasoning,
                "Combined_medication": association_result,
                "Reason": reason
            }
            
            results.append(result_row)
            
            status = "✅ YES" if association_result.lower() == 'yes' else "❌ NO"
            pbar.write(f"  ↳ CoT result: {status}")
        
        return pd.DataFrame(results)
    
    def _analyze_with_retry(self, abstract: str, drug1: str, drug2: str,
                           max_retries: int, retry_delay: int) -> Tuple[str, str]:
        """
        Analyze with retry mechanism
        
        Parameters
        ----------
        abstract : str
            Literature abstract
        drug1 : str
            First drug name
        drug2 : str
            Second drug name
        max_retries : int
            Maximum number of retry attempts
        retry_delay : int
            Retry delay
            
        Returns
        -------
        Tuple[str, str]
            (result, reasoning_process)
        """
        for attempt in range(max_retries):
            try:
                result, reasoning = self.analyze_drug_association(abstract, drug1, drug2)
                return result, reasoning
            except Exception as e:
                self.logger.error(f"CoT analysis failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return '{"result":"error", "reason":"Analysis failed"}', "Analysis failed"
        
        return '{"result":"error", "reason":"Reached maximum retry attempts"}', "Reached maximum retry attempts"
    
    def get_reasoning_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Get reasoning statistics
        
        Parameters
        ----------
        results : pd.DataFrame
            Analysis results DataFrame
            
        Returns
        -------
        Dict[str, Any]
            Statistics information
        """
        if results.empty:
            return {
                "total_papers": 0,
                "positive_associations": 0,
                "negative_associations": 0,
                "error_count": 0
            }
        
        positive = len(results[results['Combined_medication'].str.lower() == 'yes'])
        negative = len(results[results['Combined_medication'].str.lower() == 'no'])
        errors = len(results[results['Combined_medication'].str.lower() == 'error'])
        
        return {
            "total_papers": len(results),
            "positive_associations": positive,
            "negative_associations": negative,
            "error_count": errors,
            "positive_rate": positive / len(results) if len(results) > 0 else 0
        }