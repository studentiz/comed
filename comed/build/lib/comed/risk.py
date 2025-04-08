# risk.py
import pandas as pd
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from .analysis import invoke_openai_chat_completion, remove_think_tags_from_text, remove_code_block_tags

def perform_risk_analysis(content: str, model_name: str,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No") -> str:
    """
    Analyze risk factors in content using LLM.
    
    Parameters
    ----------
    content : str
        Content to analyze
    model_name : str
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    
    Returns
    -------
    str
        Risk analysis result
    """
    input_messages = [
        {"role": "system", "content": (
            "You are an excellent language master. You cannot request additional information from the user or suggest that they look for more information."
        )},
        {"role": "user", "content": content},
    ]
    
    return invoke_openai_chat_completion(
        model_name, 
        input_messages,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api
    )

def process_risk_analysis(ddc_papers_association_pd: pd.DataFrame, 
                         risk_colname: str, 
                         question_template: str, 
                         model_name: str,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No",
                         verbose: bool = True) -> pd.DataFrame:
    """
    Process risk analysis for drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str
        Column name for storing risk results
    question_template : str
        Template for risk analysis question with {entity_1} and {entity_2} placeholders
    model_name : str
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with risk analysis results
    """
    papers_count = ddc_papers_association_pd.shape[0]
    tmp_col_list = []
    tmp_col_raw_list = []

    # Define format requirements
    format_requirements = """Output the result in JSON format with two fields: "raw" for the original content and "formatted" for the transformed content. If the original content includes the phrase "No mentioned" or any variation indicating uncertainty or ambiguity (e.g., "not specified," "unclear," or "unknown"), the "formatted" field should output "Invalid". Otherwise, the "formatted" field should provide a concise summary of the original content.Example output: {"raw": "no mention", "formatted": "Invalid"}"""

    # Create progress bar with informative description
    aspect_emoji = {
        "Risks": "⚠️",
        "Safety": "🛡️",
        "Indications": "🔍",
        "Selectivity": "👥",
        "Management": "📋"
    }
    
    emoji = aspect_emoji.get(risk_colname, "📊")
    
    pbar = tqdm(range(papers_count), disable=not verbose,
                desc=f"{emoji} Analyzing {risk_colname}", 
                unit="paper")

    for i in pbar:
        tmp_item = ddc_papers_association_pd.iloc[i]
        entity_1 = tmp_item["Drug1"]
        entity_2 = tmp_item["Drug2"]
        abstract = tmp_item["Abstract"]
        
        # Update progress bar description
        pbar.set_description(f"{emoji} Analyzing {risk_colname}: {entity_1} + {entity_2} [{i+1}/{papers_count}]")

        # Construct question and content based on template
        question = question_template.format(entity_1=entity_1, entity_2=entity_2)
        content = f"""Here is an Abstract of the combined use of {entity_1} and {entity_2}. Please answer the questions based on this Abstract. **Abstract**:\n{abstract}\n**Question**:\n{question}"""

        # Perform risk analysis and process the result
        tmp_result = perform_risk_analysis(
            content, 
            model_name,
            api_key=api_key,
            api_base=api_base,
            old_openai_api=old_openai_api
        )
        
        tmp_result = remove_think_tags_from_text(tmp_result)
        
        # Format result
        formatted_result = invoke_openai_chat_completion(
            model_name,
            [
                {"role": "system", "content": "You are a formatting function. You will only output the content in the transformed format and nothing else."},
                {"role": "user", "content": f"**Content**:\n{tmp_result}\n**Formatting Requirements**:{format_requirements}"}
            ],
            api_key=api_key,
            api_base=api_base,
            old_openai_api=old_openai_api
        )
        
        formatted_result = remove_think_tags_from_text(formatted_result)
        formatted_result = remove_code_block_tags(formatted_result)

        try:
            # Convert JSON string to dictionary
            data_dict = json.loads(formatted_result)
            raw = data_dict['raw']
            formatted = data_dict['formatted']
            
            # Show result in progress bar
            status = "✅ Valid" if formatted != "Invalid" else "❌ Invalid"
            pbar.write(f"  ↳ {entity_1} + {entity_2}: {status}")
            
        except json.JSONDecodeError:
            logging.info(f"Input string is not valid JSON format.\n{formatted_result}\n{'='*50}")
            raw = "error"
            formatted = "error"
            pbar.write(f"  ↳ {entity_1} + {entity_2}: ⚠️ JSON parsing error")
            continue

        tmp_col_list.append(formatted)
        tmp_col_raw_list.append(raw)

    # Add columns to dataframe
    result_df = ddc_papers_association_pd.copy()
    result_df[risk_colname] = tmp_col_list
    result_df[risk_colname+"_raw"] = tmp_col_raw_list
    
    # Print summary
    valid_count = sum(1 for item in tmp_col_list if item != "Invalid")
    print(f"✓ {risk_colname} analysis complete: {valid_count}/{len(tmp_col_list)} valid entries")
    
    return result_df

def risk_evol_risk(ddc_papers_association_pd: pd.DataFrame, 
                  risk_colname: str = "Risks",
                  model_name: str = None,
                  api_key: Optional[str] = None,
                  api_base: Optional[str] = None,
                  old_openai_api: str = "No",
                  verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate risks of drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str, default="Risks"
        Column name for storing risk results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with risk analysis results
    """
    print("\n⚠️ Analyzing risks and side effects of drug combinations...")
    question_template = """Does the abstract mention the risks or side effects of the combined use of {entity_1} and {entity_2}? If so, please point out the relevant sentences. If not, answer "not mentioned" without any explanation."""
    
    return process_risk_analysis(
        ddc_papers_association_pd, 
        risk_colname, 
        question_template, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        verbose=verbose
    )

def risk_evol_safety(ddc_papers_association_pd: pd.DataFrame, 
                    risk_colname: str = "Safety",
                    model_name: str = None,
                    api_key: Optional[str] = None,
                    api_base: Optional[str] = None,
                    old_openai_api: str = "No",
                    verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate safety of drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str, default="Safety"
        Column name for storing safety results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with safety analysis results
    """
    print("\n🛡️ Analyzing efficacy and safety of drug combinations...")
    question_template = """Does the Abstract mention the efficacy or safety of the combination of {entity_1} and {entity_2}? If so, please point out the relevant sentences. If not, please answer "not mentioned" without any explanation."""
    
    return process_risk_analysis(
        ddc_papers_association_pd, 
        risk_colname, 
        question_template, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        verbose=verbose
    )

def risk_evol_indications(ddc_papers_association_pd: pd.DataFrame, 
                         risk_colname: str = "Indications",
                         model_name: str = None,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No",
                         verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate indications of drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str, default="Indications"
        Column name for storing indications results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with indications analysis results
    """
    print("\n🔍 Analyzing indications and selectivity of drug combinations...")
    question_template = """Does the Abstract mention the indications or selectivity of the combination of {entity_1} and {entity_2}? If so, please point out the relevant sentences. If not, please answer "not mentioned" without any explanation."""
    
    return process_risk_analysis(
        ddc_papers_association_pd, 
        risk_colname, 
        question_template, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        verbose=verbose
    )

def risk_evol_selectivity(ddc_papers_association_pd: pd.DataFrame, 
                         risk_colname: str = "Selectivity",
                         model_name: str = None,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No",
                         verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate patient selectivity for drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str, default="Selectivity"
        Column name for storing selectivity results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with selectivity analysis results
    """
    print("\n👥 Analyzing patient population and selection for drug combinations...")
    question_template = """Does the abstract mention the patient population or selection of the combination of {entity_1} and {entity_2}? If so, please point out the relevant sentences. If not, please answer "not mentioned" without any explanation."""
    
    return process_risk_analysis(
        ddc_papers_association_pd, 
        risk_colname, 
        question_template, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        verbose=verbose
    )

def risk_evol_management(ddc_papers_association_pd: pd.DataFrame, 
                        risk_colname: str = "Management",
                        model_name: str = None,
                        api_key: Optional[str] = None,
                        api_base: Optional[str] = None,
                        old_openai_api: str = "No",
                        verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate management strategies for drug combinations.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    risk_colname : str, default="Management"
        Column name for storing management results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with management analysis results
    """
    print("\n📋 Analyzing monitoring and management of drug combinations...")
    question_template = """Does the Abstract mention the monitoring and management of the combination of {entity_1} and {entity_2}? If so, please point out the relevant sentences. If not, please answer "not mentioned" without any explanation."""
    
    return process_risk_analysis(
        ddc_papers_association_pd, 
        risk_colname, 
        question_template, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        verbose=verbose
    )

def run_all_risk_evaluations(ddc_papers_association_pd: pd.DataFrame, 
                           model_name: str = None,
                           api_key: Optional[str] = None,
                           api_base: Optional[str] = None,
                           old_openai_api: str = "No",
                           verbose: bool = True) -> pd.DataFrame:
    """
    Run all risk evaluations in sequence.
    
    Parameters
    ----------
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    model_name : str, optional
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all risk analyses
    """
    # Start with base dataframe
    result_df = ddc_papers_association_pd.copy()
    
    # Run each risk evaluation in sequence
    result_df = risk_evol_risk(
        result_df, "Risks", model_name, api_key, api_base, old_openai_api, verbose
    )
    
    result_df = risk_evol_safety(
        result_df, "Safety", model_name, api_key, api_base, old_openai_api, verbose
    )
    
    result_df = risk_evol_indications(
        result_df, "Indications", model_name, api_key, api_base, old_openai_api, verbose
    )
    
    result_df = risk_evol_selectivity(
        result_df, "Selectivity", model_name, api_key, api_base, old_openai_api, verbose
    )
    
    result_df = risk_evol_management(
        result_df, "Management", model_name, api_key, api_base, old_openai_api, verbose
    )
    
    return result_df
