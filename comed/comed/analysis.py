# analysis.py
import pandas as pd
import json
import re
import logging
import difflib
import time
import requests
import openai
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Union
from tqdm import tqdm

def invoke_openai_chat_completion(model_name: str, input_messages: List[Dict[str, str]], 
                                 api_key: Optional[str] = None,
                                 api_base: Optional[str] = None,
                                 old_openai_api: str = "No",
                                 retries: int = 100, 
                                 delay: int = 3, 
                                 temperature: float = 1.0, 
                                 max_tokens: int = 3000, 
                                 top_p: float = 1.0, 
                                 timeout: int = 100) -> Optional[str]:
    """
    Invoke OpenAI API to get model completion.
    
    Parameters
    ----------
    model_name : str
        Name of the model to use
    input_messages : List[Dict[str, str]]
        List of message dictionaries for the conversation
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    retries : int, default=100
        Maximum number of retry attempts
    delay : int, default=3
        Delay between retries in seconds
    temperature : float, default=1.0
        Controls randomness in generation
    max_tokens : int, default=3000
        Maximum number of tokens to generate
    top_p : float, default=1.0
        Controls diversity via nucleus sampling
    timeout : int, default=100
        Request timeout in seconds
    
    Returns
    -------
    str or None
        Generated text from the model or None if failed
    """
    # Set API key and base if provided
    if api_key:
        openai.api_key = api_key
    if api_base:
        openai.api_base = api_base
        
    attempt = 0
    while attempt < retries:
        try:
            if old_openai_api == "Yes":
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=input_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    request_timeout=timeout
                )
                return response.choices[0].message.content
            else:
                client = OpenAI(api_key=api_key, base_url=api_base)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=input_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    timeout=timeout,
                    stream=False,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM API call failed: {e}. Attempt {attempt + 1}/{retries}")
            logging.error(f"OpenAI API call failed: {e}. Attempt {attempt + 1}/{retries}")
            attempt += 1
            if attempt < retries:
                time.sleep(delay)
            else:
                logging.error("Reached maximum retry attempts, API call failed")
                return None

# Other functions in analysis.py remain the same, just adding better progress bar descriptions

def check_drug_combinations_from_papers(ddc_pappers: pd.DataFrame, 
                                      model_name: str,
                                      api_key: Optional[str] = None,
                                      api_base: Optional[str] = None,
                                      old_openai_api: str = "No",
                                      filepath: str = "ddc_papers_association_pd.csv", 
                                      verbose: bool = True, 
                                      max_retries: int = 30, 
                                      retry_delay: int = 5) -> pd.DataFrame:
    """
    Check which papers mention drug combinations using LLM reasoning.
    
    Parameters
    ----------
    ddc_pappers : pd.DataFrame
        DataFrame of papers to analyze
    model_name : str
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    filepath : str, default="ddc_papers_association_pd.csv"
        Path to save association analysis results
    verbose : bool, default=True
        Whether to display progress bar
    max_retries : int, default=30
        Maximum number of retry attempts
    retry_delay : int, default=5
        Delay between retries in seconds
    
    Returns
    -------
    pd.DataFrame
        DataFrame with association analysis results
    """
    ddc_papers_association_pd = pd.DataFrame()

    # Create progress bar with descriptive prefix
    pbar = tqdm(range(ddc_pappers.shape[0]), disable=not verbose, 
                desc="🔍 Analyzing drug combinations in papers", 
                unit="paper")

    for i in pbar:
        tmp_item = ddc_pappers.iloc[i]
        entity_1 = tmp_item["Drug1"]
        entity_2 = tmp_item["Drug2"]
        additional_info = tmp_item["Abstract"]
        
        # Update progress bar with current paper info
        paper_title = tmp_item["Title"][:30] + "..." if len(tmp_item["Title"]) > 30 else tmp_item["Title"]
        pbar.set_description(f"🔍 Analyzing: {entity_1} + {entity_2} [{i+1}/{ddc_pappers.shape[0]}]")
        
        question = f"Does the text indicate an association between {entity_1} and {entity_2}?"

        reasoning_steps = [
            f"""**Abstract**
                {additional_info}\n
                **Question1**
                Which sentences describe the combined use of {entity_1} and {entity_2}? If there is no corresponding sentence, please honestly output "not mentioned".""",
            f"""
                **Question2**
                What do you think of the association between {entity_1} and {entity_2}? If there is no association, please output "no association".""",
            f"""
                **Question3**
                Wait, is the combined use of {entity_1} and {entity_2} being over-interpreted?""",
            f"""
                **Question4**
                Please finally determine whether the abstract mentions the combined use of {entity_1} and {entity_2}, and explain the reasons.""",
        ]

        format_description = f"""Output in JSON format with two fields: "result" and "reason". If the abstract mentions the combination of {entity_1} and {entity_2}, set "result" to "yes"; otherwise, set it to "no". Provide a detailed explanation in the "reason" field."""
        format_example = '{"result":"yes", "reason":"xxxxx"}'

        attempt = 0
        success = False
        result = "error"
        reason = "error"
        tmp_json_failed_content = "No json failed"
        reasoning = ""

        while attempt < max_retries and not success:
            try:
                # Show which reasoning step we're on
                pbar.write(f"  ↳ Analyzing paper: {paper_title}")
                
                # Call process_full_workflow to get new results and reasoning
                results, reasoning = process_full_workflow(
                    additional_info, 
                    entity_1, 
                    entity_2, 
                    question, 
                    reasoning_steps, 
                    format_description, 
                    format_example, 
                    model_name,
                    api_key=api_key,
                    api_base=api_base,
                    old_openai_api=old_openai_api
                )
                
                results = remove_code_block_tags(results)
                results = extract_json(results)
                
                # Parse JSON results
                data_dict = json.loads(results)
                result = data_dict['result']
                reason = data_dict['reason']
                success = True  # If parsing successful, exit loop
                
                # Show result
                pbar.write(f"  ↳ Result: {'✅ YES' if result.lower() == 'yes' else '❌ NO'}")

            except (requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
                logging.error(f"Network request error: {e}. Retrying attempt {attempt+1}...")
                pbar.write(f"  ↳ ⚠️ Network error, retrying ({attempt+1}/{max_retries})")
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}. Retrying attempt {attempt+1}...")
                tmp_json_failed_content = results
                pbar.write(f"  ↳ ⚠️ JSON parsing error, retrying ({attempt+1}/{max_retries})")
            except Exception as e:
                logging.error(f"Unknown error occurred: {e}. Retrying attempt {attempt+1}...")
                pbar.write(f"  ↳ ⚠️ Unknown error, retrying ({attempt+1}/{max_retries})")

            # If failed, increase attempt count and wait
            attempt += 1
            if not success and attempt < max_retries:
                time.sleep(retry_delay)
            elif not success:
                pbar.write(f"  ↳ ❌ Failed after {max_retries} retries, recording as error")
                logging.error(f"Failed after {max_retries} retries, skipping this record.")

        reasoning = clean_text(reasoning) if reasoning else ""

        # Record results
        ddc_papers_association_dic = {
            "ID": i, "Drug1": entity_1, "Drug2": entity_2,
            "PMID": tmp_item["PMID"], "Title": tmp_item["Title"], 
            "Abstract": additional_info, "Authors": tmp_item["Authors"], 
            "Journal": tmp_item["Journal"], "Publication Date": tmp_item["Publication Date"], 
            "Link": tmp_item["Link"],
            "Reasoning": reasoning, "Combined_medication": result, 
            "Json_failed": tmp_json_failed_content, "Reason": reason,
        }
        
        ddc_papers_association_pd = pd.concat(
            [ddc_papers_association_pd, pd.DataFrame([ddc_papers_association_dic])], 
            ignore_index=True
        )

        # Save incremental results
        ddc_papers_association_pd.to_csv(filepath, index=False)

    return ddc_papers_association_pd

# Include the other functions from the original analysis.py
# (remove_think_tags_from_text, remove_code_block_tags, etc.)
def remove_think_tags_from_text(text: str) -> str:
    """
    Remove <think> tags and their content from text.
    
    Parameters
    ----------
    text : str
        Input text with potential think tags
    
    Returns
    -------
    str
        Cleaned text without think tags
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def remove_code_block_tags(text: str) -> str:
    """
    Clean code block markup (like ```json ... ```) while preserving the content.
    
    Parameters
    ----------
    text : str
        Input text with potential code blocks
    
    Returns
    -------
    str
        Cleaned text without code block markers
    """
    text = re.sub(r"```.*?(\n|\s)", "", text, flags=re.DOTALL)
    text = re.sub(r"```", "", text, flags=re.DOTALL)
    return text.strip()

def remove_duplicate_content_improved(text: str, delimiter: str = "\n\n", 
                                     similarity_threshold: float = 0.85) -> str:
    """
    Improved function to remove duplicate or similar content blocks from text.
    
    Parameters
    ----------
    text : str
        Input text
    delimiter : str, default="\n\n"
        Delimiter to split text into blocks
    similarity_threshold : float, default=0.85
        Threshold to consider blocks as similar
    
    Returns
    -------
    str
        Text with duplicates removed
    """
    blocks = text.split(delimiter)
    unique_blocks = []

    for block in blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        duplicate_found = False
        for unique_block in unique_blocks:
            unique_clean = unique_block.strip()
            if block_clean in unique_clean or unique_clean in block_clean:
                duplicate_found = True
                break
            ratio = difflib.SequenceMatcher(None, block_clean, unique_clean).ratio()
            if ratio >= similarity_threshold:
                duplicate_found = True
                break

        if not duplicate_found:
            unique_blocks.append(block)

    return delimiter.join(unique_blocks)

def perform_stepwise_reasoning(reasoning_steps: List[str], additional_info: str, 
                              question: str, model_name: str,
                              api_key: Optional[str] = None,
                              api_base: Optional[str] = None,
                              old_openai_api: str = "No") -> Tuple[List[str], str]:
    """
    Perform stepwise reasoning through multiple rounds of model interaction.
    
    Parameters
    ----------
    reasoning_steps : List[str]
        List of reasoning step prompts
    additional_info : str
        Background information for reasoning
    question : str
        The question to be answered
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
    Tuple[List[str], str]
        List of reasoning results and final step result
    """
    messages = [
        {"role": "system", "content": (
            "You are an excellent language master. You cannot request additional information from the user or suggest that they look for more information."
        )},
    ]

    # Execute each step sequentially
    results = []
    tmp_no = 1
    for step in reasoning_steps:
        messages.append({"role": "user", "content": step})
        results.append(step + "\n")
        
        step_result = invoke_openai_chat_completion(
            model_name, 
            messages, 
            api_key=api_key,
            api_base=api_base,
            old_openai_api=old_openai_api
        )
        
        logging.info(f"【Reasoning】\n{step_result}\n{'-'*50}")
        results.append(f"**Answer{str(tmp_no)}**\n" + step_result + "\n")
        messages.append({"role": "assistant", "content": step_result})
        tmp_no += 1

    final_step_result = step_result

    return results, final_step_result

def perform_summary(reasoning_output: str, additional_info: str, question: str, 
                   model_name: str,
                   api_key: Optional[str] = None,
                   api_base: Optional[str] = None,
                   old_openai_api: str = "No") -> str:
    """
    Distill reasoning results into a clear conclusion.
    
    Parameters
    ----------
    reasoning_output : str
        Output from the reasoning process
    additional_info : str
        Background information
    question : str
        Original question
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
        Summarized conclusion
    """
    input_messages = [
        {"role": "system", "content": (
            "You are a summarization expert who will distill your reasoning into a final conclusion."
        )},
        {"role": "user", "content": f"**Background**:\n{additional_info}\n**Question**:\n{question}\n**Reasoning**:\n{reasoning_output}"},
    ]
    
    return invoke_openai_chat_completion(
        model_name, 
        input_messages, 
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api
    )

def format_final_output(summary_output: str, format_requirements: str, model_name: str,
                       api_key: Optional[str] = None,
                       api_base: Optional[str] = None,
                       old_openai_api: str = "No") -> Optional[str]:
    """
    Format summary output according to requirements.
    
    Parameters
    ----------
    summary_output : str
        Summary to format
    format_requirements : str
        Formatting instructions
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
    str or None
        Formatted output or None if failed
    """
    input_messages = [
        {"role": "system", "content": (
            "You are a formatting function. You will only output the content in the transformed format and nothing else."
        )},
        {"role": "user", "content": f"**Content**:\n{summary_output}\n**Formatting Requirements**:{format_requirements}"}
    ]
    
    formatted_result = invoke_openai_chat_completion(
        model_name, 
        input_messages, 
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api
    )

    if formatted_result is None:
        return None

    cleaned_result = remove_code_block_tags(formatted_result)

    try:
        json_obj = json.loads(cleaned_result)
        return json.dumps(json_obj, ensure_ascii=False)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        return cleaned_result

def process_full_workflow(additional_info: str, drug_name_1: str, drug_name_2: str, 
                         question: str, reasoning_steps: List[str], 
                         format_description: str, format_example: str, 
                         model_name: str,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No") -> Tuple[str, str]:
    """
    Execute complete workflow:
      1. Understand the problem
      2. Develop reasoning strategy
      3. Perform detailed reasoning
      4. Summarize results
      5. Format output
    
    Parameters
    ----------
    additional_info : str
        Background information
    drug_name_1 : str
        First drug name
    drug_name_2 : str
        Second drug name
    question : str
        Question to answer
    reasoning_steps : List[str]
        Reasoning steps to follow
    format_description : str
        Description of required output format
    format_example : str
        Example of output format
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
    Tuple[str, str]
        Formatted final output and cleaned reasoning output
    """
    logging.info(f"【Question to solve】\n{question}\n{'-'*50}")

    # Perform step-by-step reasoning
    reasoning_results, final_step_result = perform_stepwise_reasoning(
        reasoning_steps, 
        additional_info, 
        question, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api
    )
    logging.info(f"【Reasoning steps output】\n{reasoning_results}\n{'-'*50}")

    # Clean <think> tags
    cleaned_reasoning_output = remove_think_tags_from_text(" ".join(reasoning_results))
    cleaned_final_step_result = remove_think_tags_from_text(final_step_result)

    cleaned_summary_output = cleaned_final_step_result

    # Format output requirements
    format_requirements = f"{format_description}\nFormat example:{format_example}"
    formatted_final_output = format_final_output(
        cleaned_summary_output, 
        format_requirements, 
        model_name,
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api
    )
    
    formatted_final_output = remove_duplicate_content_improved(formatted_final_output)
    logging.info(f"【Formatted output (full output including thinking process)】\n{formatted_final_output}\n{'='*50}")
    
    # Clean <think> tags
    final_output = remove_think_tags_from_text(formatted_final_output)
    return final_output, cleaned_reasoning_output

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace.
    
    Parameters
    ----------
    text : str
        Text to clean
    
    Returns
    -------
    str
        Cleaned text
    """
    # Remove leading and trailing whitespace from each line (but keep newlines)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
    # Replace multiple consecutive spaces with a single space (preserve newlines)
    text = re.sub(r' +', ' ', text)
    return text

def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON data from text.
    
    Parameters
    ----------
    text : str
        Text potentially containing JSON
    
    Returns
    -------
    str or None
        Extracted JSON string or None if not found
    """
    # Use regex to find potential JSON format
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            # Try to parse the extracted string as JSON
            json_data = json.loads(match.group(0))
            # Return JSON string with double quotes
            return json.dumps(json_data, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, return None
            return None
    return None

def format_reasoning_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format reasoning outputs for better readability.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with reasoning data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with formatted reasoning
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if 'Reasoning' not in result_df.columns:
        return result_df
    
    print("📝 Formatting reasoning outputs for better readability...")
    
    new_reasoning_list = []
    for reasoning in result_df["Reasoning"]:
        tmp_str = str(reasoning)
        
        # Fix formatting issues
        tmp_str = tmp_str.replace("""association, please output "no association """, """association, please output "no association".\n""")
        tmp_str = tmp_str.replace('output "no association', 'output "no association".')
        tmp_str = tmp_str.replace('?\nIf there is no association', '? If there is no association')
        tmp_str = tmp_str.replace('""', '"')
        
        # Ensure answer tags are present
        if "**Answer1**" not in tmp_str:
            tmp_str = tmp_str.replace("""corresponding sentence, please honestly output "not mentioned".\n""", 
                                     """corresponding sentence, please honestly output "not mentioned".\n**Answer1**\n""")
            tmp_str = tmp_str.replace("""corresponding sentence, please honestly output "not mentioned". """, 
                                     """corresponding sentence, please honestly output "not mentioned".\n**Answer1**\n""")
            
        if "**Answer2**" not in tmp_str:
            tmp_str = tmp_str.replace("""association, please output "no association".\n""", 
                                     """association, please output "no association".\n**Answer2**\n""")
            tmp_str = tmp_str.replace("""association, please output "no association". """, 
                                     """association, please output "no association".\n**Answer2**\n""")
            
        if "**Answer3**" not in tmp_str:
            tmp_str = tmp_str.replace("""being over-interpreted?\n""", 
                                     """being over-interpreted?\n**Answer3**\n""")
            tmp_str = tmp_str.replace("""being over-interpreted? """, 
                                     """being over-interpreted?\n**Answer3**\n""")
            
        if "**Answer4**" not in tmp_str:
            tmp_str = tmp_str.replace(""", and explain the reasons.\n""", 
                                     """, and explain the reasons.\n**Answer4**\n""")
            tmp_str = tmp_str.replace(""", and explain the reasons. """, 
                                     """, and explain the reasons.\n**Answer4**\n""")
        
        new_reasoning_list.append(tmp_str)
    
    result_df["Reasoning"] = new_reasoning_list
    print("✓ Reasoning outputs formatted successfully")
    return result_df
