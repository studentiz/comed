# report.py
import pandas as pd
import time
import os
from string import Template
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
from .analysis import invoke_openai_chat_completion

def generate_reference_no_and_tmpdf(entity_1: str, entity_2: str, 
                                  ddc_papers_association_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataframe with numbered references for a drug pair.
    
    Parameters
    ----------
    entity_1 : str
        First drug name
    entity_2 : str
        Second drug name
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association results
    
    Returns
    -------
    pd.DataFrame
        DataFrame with reference numbers
    """
    df = ddc_papers_association_pd[
        (ddc_papers_association_pd['Drug1'].str.contains(entity_1, case=False, na=False)) &
        (ddc_papers_association_pd['Drug2'].str.contains(entity_2, case=False, na=False)) &
        (ddc_papers_association_pd['Combined_medication'].str.contains('yes', case=False, na=False))
    ].copy()

    # Filter for rows with at least one non-'Invalid' value
    valid_mask = (
        (df['Risks'] != 'Invalid') |
        (df['Safety'] != 'Invalid') |
        (df['Indications'] != 'Invalid') |
        (df['Selectivity'] != 'Invalid') |
        (df['Management'] != 'Invalid')
    )

    valid_df = df[valid_mask].copy()

    # Add sequential numbering
    valid_df['Ref_NO'] = range(1, len(valid_df) + 1)

    # Return results
    return valid_df.reset_index()

def summarize_specific_risk(focus: str, entity_1: str, entity_2: str, 
                          ddc_papers_association_pd: pd.DataFrame,
                          model_name: str,
                          api_key: Optional[str] = None,
                          api_base: Optional[str] = None,
                          old_openai_api: str = "No") -> str:
    """
    Summarize specific risk aspect for a drug pair.
    
    Parameters
    ----------
    focus : str
        Risk aspect to focus on
    entity_1 : str
        First drug name
    entity_2 : str
        Second drug name
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association and risk results
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
        Summarized risk information
    """
    tmp_pd = ddc_papers_association_pd.query(f"Drug1=='{entity_1}' and Drug2=='{entity_2}' and {focus}!='Invalid'").copy()
    tmp_paper_count = tmp_pd.shape[0]
    
    if tmp_paper_count <= 0:
        return "No relevant reports were found in the searched paper abstracts."

    tmp_references_str = ""

    for i in range(tmp_paper_count):
        tmp_item = tmp_pd.iloc[i]
        ref_no = tmp_item["Ref_NO"]
        focus_content = tmp_item[focus]
        tmp_references_str += f"Ref_{ref_no}:" + focus_content + "\n"

    input_messages = [
        {"role": "system", "content": (
            "You are a clinical pharmacology expert who specializes in summarizing information."
        )},
        {"role": "user", "content": f'''
        Please summarize the risks or side effects of combining {entity_1} and {entity_2} based on the following {tmp_paper_count} references.
        **References**:
        {tmp_references_str}\n
        **Requirement**:
        Please summarize the information provided above into a single paragraph, detailing the combination of {entity_1} and {entity_2}, and mark the reference numbers of the related papers.
        '''}
    ]

    tmp_report_str = invoke_openai_chat_completion(
        model_name, 
        input_messages, 
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        temperature=0
    )
    
    return tmp_report_str

def summarize_overall_risk(entity_1: str, entity_2: str, 
                         ddc_papers_association_pd: pd.DataFrame,
                         model_name: str,
                         api_key: Optional[str] = None,
                         api_base: Optional[str] = None,
                         old_openai_api: str = "No") -> tuple:
    """
    Generate overall risk summary for a drug pair.
    
    Parameters
    ----------
    entity_1 : str
        First drug name
    entity_2 : str
        Second drug name
    ddc_papers_association_pd : pd.DataFrame
        DataFrame with association and risk results
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
    tuple
        Tuple of (overall_summary, risks, safety, indications, selectivity, management)
    """
    print(f"  ↳ Generating risk summary for {entity_1} + {entity_2}...")
    
    # Summarize each risk aspect
    print(f"    • Analyzing risks and side effects")
    risks_content = summarize_specific_risk(
        "Risks", entity_1, entity_2, ddc_papers_association_pd, 
        model_name, api_key, api_base, old_openai_api
    )
    
    print(f"    • Analyzing efficacy and safety")
    safety_content = summarize_specific_risk(
        "Safety", entity_1, entity_2, ddc_papers_association_pd,
        model_name, api_key, api_base, old_openai_api
    )
    
    print(f"    • Analyzing indications and selectivity")
    indications_content = summarize_specific_risk(
        "Indications", entity_1, entity_2, ddc_papers_association_pd,
        model_name, api_key, api_base, old_openai_api
    )
    
    print(f"    • Analyzing patient population and selection")
    selectivity_content = summarize_specific_risk(
        "Selectivity", entity_1, entity_2, ddc_papers_association_pd,
        model_name, api_key, api_base, old_openai_api
    )
    
    print(f"    • Analyzing monitoring and management")
    management_content = summarize_specific_risk(
        "Management", entity_1, entity_2, ddc_papers_association_pd,
        model_name, api_key, api_base, old_openai_api
    )

    # Combine all valid summaries
    summary_content = ""
    if risks_content != "No relevant reports were found in the searched paper abstracts.":
        summary_content += f"The risks and side effects of the combined use of {entity_1} and {entity_2} are as follows:\n" + risks_content + "\n"
    if safety_content != "No relevant reports were found in the searched paper abstracts.":
        summary_content += f"The safety of the combined use of {entity_1} and {entity_2} are as follows:\n" + safety_content + "\n"
    if indications_content != "No relevant reports were found in the searched paper abstracts.":
        summary_content += f"The indications of the combined use of {entity_1} and {entity_2} are as follows:\n" + indications_content + "\n"
    if selectivity_content != "No relevant reports were found in the searched paper abstracts.":
        summary_content += f"The selectivity of the combined use of {entity_1} and {entity_2} are as follows:\n" + selectivity_content + "\n"
    if management_content != "No relevant reports were found in the searched paper abstracts.":
        summary_content += f"The management of the combined use of {entity_1} and {entity_2} are as follows:\n" + management_content + "\n"

    # If no valid summaries found
    if summary_content == "":
        print(f"    ❌ No valid information found for {entity_1} + {entity_2}")
        return ("No relevant reports were found in the searched paper abstracts.",
                "No relevant reports were found in the searched paper abstracts.",
                "No relevant reports were found in the searched paper abstracts.",
                "No relevant reports were found in the searched paper abstracts.",
                "No relevant reports were found in the searched paper abstracts.",
                "No relevant reports were found in the searched paper abstracts.")

    # Generate overall summary
    print(f"    • Generating comprehensive summary")
    input_messages = [
        {"role": "system", "content": (
            "You are a clinical pharmacology expert who specializes in summarizing information."
        )},
        {"role": "user", "content": f'''
        Please summarize the risks of combining {entity_1} and {entity_2} based on the following information.
        **Information**:
        {summary_content}\n
        **Requirement**:
        Please summarize the information provided above in one paragraph and keep the reference identified.
        '''}
    ]

    overall_risk_content = invoke_openai_chat_completion(
        model_name, 
        input_messages, 
        api_key=api_key,
        api_base=api_base,
        old_openai_api=old_openai_api,
        temperature=0
    )
    
    print(f"    ✓ Summary complete for {entity_1} + {entity_2}")
    
    return (overall_risk_content, risks_content, safety_content, 
            indications_content, selectivity_content, management_content)

def generate_references2html(tmp_pd: pd.DataFrame) -> str:
    """
    Generate HTML for references section.
    
    Parameters
    ----------
    tmp_pd : pd.DataFrame
        DataFrame with paper references
    
    Returns
    -------
    str
        HTML string for references section
    """
    tmp_paper_count = tmp_pd.shape[0]
    if tmp_paper_count <= 0:
        return """<p class="references">No relevant papers were found.</p>"""

    tmp_references_str = ""

    for i in range(tmp_paper_count):
        tmp_item = tmp_pd.iloc[i]
        ref_no = tmp_item["Ref_NO"]
        link = tmp_item["Link"]
        title = tmp_item["Title"]
        tmp_references_str += f"""<p class="references">{ref_no}. <a href="{link}">{title}</a></p>""" + "\n"

    return tmp_references_str

def generate_html_report(ddc_papers_risk: pd.DataFrame,
                       model_name: str,
                       api_key: Optional[str] = None,
                       api_base: Optional[str] = None,
                       old_openai_api: str = "No",
                       output_file: str = "CoMed_Risk_Analysis_Report.html",
                       verbose: bool = True) -> str:
    """
    Generate HTML report for risk analysis.
    
    Parameters
    ----------
    ddc_papers_risk : pd.DataFrame
        DataFrame with risk analysis results
    model_name : str
        Name of the model to use
    api_key : str, optional
        OpenAI API key
    api_base : str, optional
        Base URL for API
    old_openai_api : str, default="No"
        Whether to use old OpenAI API format
    output_file : str, default="CoMed_Risk_Analysis_Report.html"
        Path to save HTML report
    verbose : bool, default=True
        Whether to display progress bar
    
    Returns
    -------
    str
        Path to generated HTML report
    """
    # Get unique drug combinations
    drugs_combinations = []
    unique_combinations = pd.concat([
        ddc_papers_risk[['Drug1', 'Drug2']].drop_duplicates()
    ]).values.tolist()
    
    for combo in unique_combinations:
        if combo not in drugs_combinations:
            drugs_combinations.append(combo)
    
    # Generate HTML content for each drug combination
    drugs_combination_counts = len(drugs_combinations)
    html_content = ""
    
    # Create progress bar with descriptive prefix
    pbar = tqdm(range(drugs_combination_counts), disable=not verbose,
               desc="📝 Generating report sections", 
               unit="combo")
    
    for i in pbar:
        combo = drugs_combinations[i]
        entity_1 = combo[0]
        entity_2 = combo[1]
        
        # Update progress bar description
        pbar.set_description(f"📝 Creating report for {entity_1} + {entity_2} [{i+1}/{drugs_combination_counts}]")

        tmp_df = generate_reference_no_and_tmpdf(entity_1, entity_2, ddc_papers_risk)
        
        reference_count = len(tmp_df)
        pbar.write(f"  ↳ Processing {entity_1} + {entity_2} with {reference_count} references")
        
        overall_risk_content, risks_content, safety_content, indications_content, selectivity_content, management_content = summarize_overall_risk(
            entity_1, entity_2, tmp_df, model_name, api_key, api_base, old_openai_api
        )

        references_content = generate_references2html(tmp_df)

        html_content += f"""
        <h2>Combination of {entity_1} and {entity_2}</h2>

        <div class="drug-pair">
            <h3><span style="color: red;">Overall Assessment</span></h3>
            <p>{overall_risk_content}</p>
        </div>

        <div class="drug-pair">
            <h3><span style="color: orange;">Combination therapy and side effects</span></h3>
            <p>{risks_content}</p>
        </div>

        <div class="drug-pair">
            <h3><span style="color: orange;">Efficacy and safety</span></h3>
            <p>{safety_content}</p>
        </div>

        <div class="drug-pair">
            <h3><span style="color: orange;">Indications or selectivity</span></h3>
            <p>{indications_content}</p>
        </div>

        <div class="drug-pair">
            <h3><span style="color: orange;">Patient population and selection</span></h3>
            <p>{selectivity_content}</p>
        </div>

        <div class="drug-pair">
            <h3><span style="color: orange;">Monitoring and management</span></h3>
            <p>{management_content}</p>
        </div>

         <div class="drug-pair">
            <h3><span style="color: rgb(56, 54, 50);">References</span></h3>
            {references_content}
        </div>
        """

    # Prepare metadata
    print("📊 Compiling report metadata and finalizing HTML...")
    report_created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    llm_version = model_name
    drug_combinations_no = str(len(drugs_combinations))
    papers_evaluated_no = str(ddc_papers_risk.shape[0])
    related_no = str(ddc_papers_risk[ddc_papers_risk['Combined_medication'].str.lower() == 'yes'].shape[0])

    # Count papers with valid risk information
    columns_to_check = ['Risks', 'Safety', 'Indications', 'Selectivity', 'Management']
    valid_no = str((~(ddc_papers_risk[columns_to_check] == 'Invalid').all(axis=1)).sum())

    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CoMed Risk Analysis Report</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                color: #333;
                line-height: 1.6;
            }
            .container {
                width: 75%;
                margin: auto;
                padding: 20px;
                background-color: #fff;
            }
            .container p {
                text-align: justify;
                margin-bottom: -2px;
            }
            h1 {
                text-align: center;
                color: #0056b3;
                padding-bottom: 20px;
                border-bottom: 2px solid #0056b3;
            }
            h2 {
                color: #0056b3;
                margin-top: 5px;
                margin-bottom: 5px;
            }
            h3 {
                color: #555;
                margin-top: -5px;
                margin-bottom: -15px;
            }
            .drug-pair {
                margin-bottom: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .risk-assessment {
                font-weight: bold;
                color: #d9534f;
            }
            .references {
                font-size: 0.9em;
                color: #777;
            }
            .report-metadata {
                margin-bottom: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 10px;
            }
            .metadata-item {
            }
            @media (max-width: 768px) {
                .report-metadata {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CoMed Risk Analysis Report</h1>
            <!-- metadata -->
            <div class="report-metadata">
                <div class="metadata-item">
                    <strong>Report Created:</strong> $report_created_time
                </div>
                <div class="metadata-item">
                    <strong>LLM Version:</strong> $llm_version
                </div>
                <div class="metadata-item">
                    <strong>Total number of drug combinations:</strong> $drug_combinations_no
                </div>
                <div class="metadata-item">
                    <strong>Total number of papers evaluated:</strong> $papers_evaluated_no
                </div>
                <div class="metadata-item">
                    <strong>Total number of related papers:</strong> $related_no
                </div>
                <div class="metadata-item">
                    <strong>Total number of valid papers:</strong> $valid_no
                </div>
            </div>
            $html_content
        </div>
    </body>
    </html>
    """

    # Define variables to insert
    data = {
        "report_created_time": report_created_time,
        "llm_version": llm_version,
        "drug_combinations_no": drug_combinations_no,
        "papers_evaluated_no": papers_evaluated_no,
        "related_no": related_no,
        "valid_no": valid_no,
        "html_content": html_content
    }

    # Create Template object
    template = Template(html_template)

    # Use substitute method to insert variables
    html_page = template.substitute(data)

    # Save HTML file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(html_page)

    print(f"✅ HTML report saved to {output_file}")
    return output_file
