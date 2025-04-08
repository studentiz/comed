[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI version](https://badge.fury.io/py/comed.svg?icon=si%3Apython)](https://badge.fury.io/py/comed)
# CoMed: A Framework for Drug Co-Medication Risk Analysis
CoMed is a comprehensive framework for analyzing drug co-medication risks using Chain-of-Thought (CoT) reasoning with large language models. It automates the process of searching medical literature, analyzing drug interactions, and generating detailed risk assessment reports for healthcare professionals and researchers.
## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Understanding the Output](#-understanding-the-output)
- [API Reference](#-api-reference)
- [Advanced Configuration](#-advanced-configuration)
- [Example Use Cases](#-example-use-cases)
- [How It Works](#-how-it-works)
- [Customization](#-customization)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [FAQ](#-faq)
## ✨ Features
| Feature | Description |
|---------|-------------|
| 🔍 **Literature Search** | Automatically search PubMed for relevant papers discussing specific drug combinations |
| 🧠 **Chain-of-Thought Analysis** | Use LLM-based structured reasoning to identify true drug combination mentions, filtering out papers that merely mention both drugs separately |
| ⚠️ **Comprehensive Risk Assessment** | Evaluate multiple dimensions of drug interaction risks including side effects, efficacy, indications, patient selection, and management |
| 📊 **Interactive Reporting** | Generate interactive HTML reports with rich information and direct links to source literature |
| 🔄 **Incremental Analysis** | Support for adding drugs incrementally to existing analyses, saving computational resources |
| 📁 **Data Persistence** | Save intermediate results at each step, allowing for workflow interruption and resumption |
## 📥 Installation
### Prerequisites
- Python 3.12 or higher
- Access to an LLM API (OpenAI, Claude, Qwen, etc.)
### Via PyPI (Recommended)
```bash
pip install comed
```
### From Source (Latest Development Version)
```bash
git clone https://github.com/username/comed.git
cd comed
pip install -e .
```
### Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| pandas | >=2.2.3 | Data manipulation and analysis |
| numpy | >=2.2.3 | Numerical computing |
| biopython | >=1.85 | Interface with biological databases including PubMed |
| tqdm | >=4.67.1 | Progress bar visualization |
| openai | >=1.65.1 | OpenAI API client |
| requests | >=2.32.3 | HTTP requests |
| typing-extensions | >=4.7.0 | Type hinting extensions |
## 🚀 Quick Start
### Configuration
```python
import os
import comed
# Configure your LLM API credentials
os.environ["MODEL_NAME"] = "gpt-4"  # or any compatible model
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key"
```
### Basic Usage
```python
# Initialize CoMed with a list of drugs
drugs = ["warfarin", "aspirin", "ibuprofen", "clopidogrel"]
com = comed.CoMedData(drugs)
# Run the full analysis pipeline with detailed progress information
com.run_full_analysis(retmax=30, verbose=True)
print(f"Report generated at: {com.report_path}")
```
### Step-by-Step Analysis
```python
import comed
import logging
# Configure logging to see detailed progress
comed.utils.configure_logging(log_level=logging.INFO)
# Initialize with drug list
drugs = ["metformin", "lisinopril", "atorvastatin"]
com = comed.CoMedData(drugs)
# Step 1: Search medical literature
# This will find papers that mention each drug combination
com.search(retmax=30, email="your.email@example.com")
print(f"Found {len(com.papers)} papers mentioning the drug combinations")
# Step 2: Analyze which papers actually discuss drug combinations
com.analyze_associations()
positive_count = len(com.associations[com.associations['Combined_medication'].str.lower() == 'yes'])
print(f"Found {positive_count} papers with relevant drug combinations")
# Step 3: Analyze different risk dimensions
com.analyze_risks()
print("Risk analysis complete")
# Step 4: Generate HTML report
report_path = com.generate_report("drug_interactions.html")
print(f"Report saved to: {report_path}")
```
### Using Different LLM Models
```python
import comed
# Initialize with drugs
com = comed.CoMedData(["simvastatin", "amlodipine"])
# Configure for specific LLM (QianWen example)
com.set_config({
    'model_name': 'qwen-max',
    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'your-dashscope-api-key'
})
# Run analysis with verbose output
com.run_full_analysis(retmax=20, verbose=True)
```
### Working with Existing Data
```python
import comed
import comed.io as io
# Load data from previous analysis
papers = io.read_papers("ddc_papers.csv")
associations = io.read_associations("ddc_papers_association_pd.csv")
risk_analysis = io.read_risk_analysis("ddc_papers_risk.csv")
# Create CoMed instance with loaded data
com = comed.CoMedData()
com.papers = papers
com.associations = associations
com.risk_analysis = risk_analysis
# Generate new report from existing data
com.generate_report("updated_report.html")
```
### Adding More Drugs to Analysis
```python
import comed
# Start with a few drugs
com = comed.CoMedData(["warfarin", "aspirin"])
com.search(retmax=30)
# Add more drugs later
com.add_drugs(["heparin", "clopidogrel"])
# Only searches the new combinations
com.search(retmax=30)
# Complete the analysis with method chaining
com.analyze_associations() \
   .analyze_risks() \
   .generate_report("anticoagulant_interactions.html")
```
## 📊 Understanding the Output
CoMed generates a comprehensive HTML report with multiple sections:
### Report Structure
| Section | Content |
|---------|---------|
| **Meta Information** | Analysis timestamp, LLM model used, drug combinations analyzed, paper counts |
| **Drug Combination Sections** | Detailed analysis for each drug pair |
| **Overall Assessment** | Overall risk evaluation with evidence |
| **Side Effects** | Potential adverse events when drugs are combined |
| **Efficacy & Safety** | How the combination affects therapeutic outcomes |
| **Indications & Contraindications** | When the combination is appropriate or should be avoided |
| **Patient Selection** | Which patient populations may benefit or face increased risks |
| **Clinical Management** | Recommendations for monitoring and managing the combination |
| **References** | Numbered links to source literature on PubMed |
### Sample Report Output
Each drug combination section includes detailed information about the clinical implications of co-administering the medications, drawn directly from the medical literature:
```
Combination of warfarin and aspirin
Overall Assessment:
The combination of warfarin and aspirin is associated with a significantly increased risk 
of bleeding complications (Ref_1, Ref_3, Ref_4). While this combination may be necessary 
in certain clinical scenarios such as patients with mechanical heart valves and acute 
coronary syndrome (Ref_2), careful monitoring of INR levels and bleeding signs is essential. 
The therapeutic benefits must be weighed against the heightened bleeding risk, and the 
duration of combination therapy should be minimized whenever possible.
Combination therapy and side effects:
Combined use of warfarin and aspirin significantly increases the risk of major bleeding 
events, with studies showing a 2-3 fold increase in bleeding complications compared to 
either agent alone (Ref_1, Ref_4). Gastrointestinal bleeding is particularly common, 
and intracranial hemorrhage, while rare, represents a serious concern (Ref_3).
...
```
## 📚 API Reference
### CoMedData Class
The core class that handles the entire analysis pipeline.
| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `__init__(drugs=None)` | Initialize CoMedData with optional drug list | `drugs`: list of drug names | `CoMedData` instance |
| `set_config(config)` | Configure LLM API settings | `config`: dict with model_name, api_base, api_key, etc. | `self` (chainable) |
| `add_drugs(drugs)` | Add more drugs to existing analysis | `drugs`: list of new drug names | `self` (chainable) |
| `search(retmax=30, email='your_email@example.com', retry=3, delay=3, filepath="ddc_papers.csv", verbose=True)` | Search PubMed for papers | `retmax`: max papers per drug pair<br>`email`: PubMed contact<br>`retry`: max retries<br>`delay`: seconds between retries<br>`filepath`: where to save results<br>`verbose`: show progress | `self` (chainable) |
| `analyze_associations(filepath="ddc_papers_association_pd.csv", verbose=True, max_retries=30, retry_delay=5)` | Analyze drug combination mentions | `filepath`: where to save results<br>`verbose`: show progress<br>`max_retries`: API failure retries<br>`retry_delay`: seconds between retries | `self` (chainable) |
| `analyze_risks(filepath="ddc_papers_risk.csv", verbose=True)` | Analyze risk dimensions | `filepath`: where to save results<br>`verbose`: show progress | `self` (chainable) |
| `generate_report(output_file="CoMed_Risk_Analysis_Report.html", verbose=True)` | Generate HTML report | `output_file`: report filename<br>`verbose`: show progress | Report path (str) |
| `run_full_analysis(retmax=30, verbose=True)` | Run entire analysis pipeline | `retmax`: max papers per drug pair<br>`verbose`: show progress | Report path (str) |
### IO Functions
Functions to load and save analysis data.
| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `read_papers(filepath)` | Load paper data | `filepath`: path to CSV file | DataFrame |
| `read_associations(filepath)` | Load association data | `filepath`: path to CSV file | DataFrame |
| `read_risk_analysis(filepath)` | Load risk analysis data | `filepath`: path to CSV file | DataFrame |
| `save_dataframe(df, filepath, index=False)` | Save DataFrame to CSV | `df`: DataFrame to save<br>`filepath`: output path<br>`index`: include index | None |
## ⚙️ Advanced Configuration
### Environment Variables
| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MODEL_NAME` | LLM model to use | None | `"gpt-4"` |
| `API_BASE` | Base URL for API | None | `"https://api.openai.com/v1"` |
| `API_KEY` | API authentication key | None | `"sk-..."` |
| `LOG_DIR` | Directory for log files | `"logs"` | `"my_logs"` |
| `OLD_OPENAI_API` | Use legacy OpenAI API | `"No"` | `"Yes"` |
### Logging Configuration
```python
import logging
import comed.utils as utils
# Configure detailed logging to file
utils.configure_logging(
    log_file="comed_analysis.log", 
    log_level=logging.DEBUG,
    console_level=logging.INFO  # Show INFO and above in console
)
```
### Progress Display Options
```python
import comed
drugs = ["warfarin", "aspirin", "clopidogrel"]
com = comed.CoMedData(drugs)
# Enable detailed progress information for each step
com.search(retmax=30, verbose=True)
com.analyze_associations(verbose=True)
com.analyze_risks(verbose=True)
com.generate_report(verbose=True)
# Or disable progress display for batch processing
com.run_full_analysis(retmax=30, verbose=False)
```
## 🔬 Example Use Cases
### Case 1: Analyzing Common Drug Combinations in Elderly Patients
```python
import comed
import logging
# Set up informative logging
comed.utils.configure_logging(log_level=logging.INFO)
# Common medications prescribed to elderly patients
drugs = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "levothyroxine"]
com = comed.CoMedData(drugs)
# Run analysis with detailed console output
report = com.run_full_analysis(retmax=50)
print(f"Elderly medication interactions analysis complete. Report at: {report}")
```
### Case 2: Focused Analysis on Anticoagulant Interactions
```python
import comed
import pandas as pd
# Focus on anticoagulants and common co-medications
drugs = ["warfarin", "apixaban", "rivaroxaban", "clopidogrel", "aspirin"]
com = comed.CoMedData(drugs)
# Detailed step-by-step approach
com.search(retmax=50)
print(f"Found {len(com.papers)} papers mentioning the drug combinations")
# Filter papers by publication date (last 3 years)
current_year = pd.Timestamp.now().year
recent_papers = com.papers[com.papers["Publication Date"].str.contains(f"{current_year}|{current_year-1}|{current_year-2}")]
print(f"Filtered to {len(recent_papers)} recent papers")
com.papers = recent_papers
# Continue analysis with filtered papers
com.analyze_associations() \
   .analyze_risks() \
   .generate_report("anticoagulant_interactions.html")
```
### Case 3: Batch Analysis of Multiple Drug Classes
```python
import comed
import os
# Define drug classes
antihypertensives = ["lisinopril", "amlodipine", "hydrochlorothiazide"]
antidiabetics = ["metformin", "sitagliptin", "insulin"]
statins = ["atorvastatin", "simvastatin", "rosuvastatin"]
# Create output directory
os.makedirs("drug_interaction_reports", exist_ok=True)
# Analysis 1: Antihypertensives with antidiabetics
com1 = comed.CoMedData(antihypertensives + antidiabetics)
report1 = com1.run_full_analysis(retmax=20)
print(f"Analysis 1 complete: {report1}")
# Analysis 2: Antidiabetics with statins
com2 = comed.CoMedData(antidiabetics + statins)
report2 = com2.run_full_analysis(retmax=20)
print(f"Analysis 2 complete: {report2}")
# Analysis 3: Antihypertensives with statins
com3 = comed.CoMedData(antihypertensives + statins)
report3 = com3.run_full_analysis(retmax=20)
print(f"Analysis 3 complete: {report3}")
```
## 🔄 How It Works
CoMed uses a four-step process to analyze co-medication risks:
### 1. Literature Search
For each drug pair, CoMed queries PubMed to find relevant medical papers that mention both drugs. The search uses precise query construction to maximize relevance:
```
{drug1}[Title/Abstract] AND {drug2}[Title/Abstract]
```
Papers are filtered to ensure they contain proper abstracts and metadata. Each paper's PMID, title, abstract, authors, journal, publication date, and link are stored for further analysis.
### 2. Association Analysis
Using Chain-of-Thought reasoning, each paper is analyzed by the LLM to determine if it genuinely discusses the combined use of the drugs (not just mentioning them separately). This multi-step reasoning process includes:
1. Identifying specific sentences that mention combined use
2. Determining the nature of the association
3. Cross-checking for potential over-interpretation
4. Making a final determination with justification
This ensures that only papers truly discussing drug combinations are included in the analysis.
### 3. Risk Evaluation
For papers with confirmed drug combinations, CoMed analyzes five key dimensions:
| Dimension | Analysis Focus |
|-----------|---------------|
| **Adverse events & Side Effects** | Adverse events, toxicities, and complications |
| **Efficacy & Safety** | Therapeutic outcomes and general safety profile |
| **Indications & Contraindications** | Clinical scenarios where combination is appropriate or contraindicated |
| **Patient Selection** | Patient characteristics affecting risk-benefit profile |
| **Monitoring & Management** | Strategies for safe co-administration and complication prevention |
### 4. Report Generation
CoMed summarizes the findings into a comprehensive HTML report, providing clinically relevant information with references to the source literature. The report is structured for easy navigation and includes:
- Meta-information about the analysis
- Drug-by-drug combination analysis
- Detailed risk assessments across all dimensions
- Direct links to source literature for verification
## 🛠️ Customization
### Supporting Different APIs
To use alternative LLM APIs, set the relevant configuration:
```python
# For Anthropic Claude
com = comed.CoMedData(drugs)
com.set_config({
    'model_name': 'claude-3-opus-20240229',
    'api_base': 'https://api.anthropic.com/v1',
    'api_key': 'your-anthropic-key',
    'old_openai_api': 'No'
})
# For QianWen
com = comed.CoMedData(drugs)
com.set_config({
    'model_name': 'qwen-max',
    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'your-dashscope-key',
    'old_openai_api': 'No'
})
```
### LLM Model Comparison
| Model | Compatibility | Quality | Speed | Cost Efficiency |
|-------|---------------|---------|-------|----------------|
| GPT-4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Claude 3 Opus | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Qwen Max | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Llama 3 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
### Customizing Analysis Parameters
```python
import comed
# Create instance with custom configuration
com = comed.CoMedData(["warfarin", "aspirin"])
# Configure chain-of-thought steps for association analysis
com.association_reasoning_steps = [
    "What sentences explicitly mention combined use of {entity_1} and {entity_2}?",
    "Is there evidence of a pharmacokinetic or pharmacodynamic interaction?",
    "What does the literature conclude about this drug combination?",
    "Final determination: Does this paper discuss combined use?"
]
# Run with custom parameters
com.search(retmax=50)
com.analyze_associations(max_retries=50, retry_delay=10)
com.analyze_risks()
com.generate_report()
```
## 👥 Contributing
We welcome contributions to CoMed! Here's how you can help:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## ❓ FAQ
### Q: How many drugs can I analyze at once?
**A:** CoMed can analyze any number of drugs, but the number of combinations grows quickly (n*(n-1)/2). For larger analyses, consider breaking it into smaller drug groups to manage computational resources. A typical analysis of 5 drugs (10 combinations) takes approximately 1-2 hours depending on the LLM used.
### Q: Which LLMs work best with CoMed?
**A:** CoMed performs best with more advanced models like GPT-4, Claude 3 Opus, or Qwen Max, but will work with any model that can handle chain-of-thought reasoning. More capable models produce higher quality medical analyses and are recommended for clinical research purposes.
### Q: How can I interpret the risk levels in the report?
**A:** CoMed summarizes findings from the literature but does not assign specific risk levels. The reports include the context and evidence from medical papers, which should be evaluated by healthcare professionals based on:
- Frequency of reported adverse events
- Severity of potential interactions
- Consistency across multiple papers
- Quality of the evidence (study design, sample size)
### Q: How recent are the papers CoMed analyzes?
**A:** CoMed searches PubMed for the most relevant papers based on your query terms. By default, results are sorted by relevance rather than date. You can filter papers by date in post-processing if you need more recent literature, as shown in Example Case 2.
### Q: Is CoMed suitable for clinical decision-making?
**A:** CoMed is designed as a research tool and information aggregator. Any information used for clinical decisions should be verified by qualified healthcare professionals. The tool helps identify potential interactions and summarizes literature, but does not replace clinical judgment or established drug interaction databases.
### Q: Can I analyze drug interactions with specific conditions or diseases?
**A:** While CoMed is designed primarily for drug-drug interactions, you can include disease terms in your search by customizing the search queries programmatically. Contact the developers for guidance on adapting CoMed for specific research needs.
### Q: How does CoMed handle different drug names and formulations?
**A:** CoMed searches using the exact drug names provided. For comprehensive analysis, consider including both generic and brand names, as well as different salt forms if relevant. Future versions may include automatic synonym expansion.
