# CoMed 🧬💊
### A Framework for Analyzing Co-Medication Risks using Chain-of-Thought Reasoning
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Documentation Status](https://readthedocs.org/projects/comed/badge/?version=latest)](https://comed.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/username/comed)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg)](https://doi.org/10.5281/zenodo.12345678)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/username/comed/blob/main/CONTRIBUTING.md)
<p align="center">
  <img src="https://github.com/username/comed/raw/main/docs/images/comed_logo.png" alt="CoMed Logo" width="300"/>
</p>
CoMed is a comprehensive framework for analyzing drug co-medication risks using Chain-of-Thought (CoT) reasoning and large language models. It automates the process of searching medical literature, analyzing drug interactions, and generating detailed risk assessment reports.
## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Understanding the Pipeline](#-understanding-the-pipeline)
- [API Reference](#-api-reference)
- [Advanced Configuration](#-advanced-configuration)
- [Real-World Examples](#-real-world-examples)
- [Performance Benchmarks](#-performance-benchmarks)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)
- [FAQ](#-faq)
## 🔭 Overview
Drug-drug interactions are a major cause of adverse drug events (ADEs) and represent a significant clinical and public health challenge. CoMed automates the analysis of co-medication risks by:
1. **Searching scientific literature** for mentions of specific drug combinations
2. **Applying Chain-of-Thought reasoning** to determine genuine co-medication discussions
3. **Analyzing multiple dimensions of risk** through a systematic assessment framework
4. **Generating comprehensive reports** with evidence-backed insights
<p align="center">
  <img src="https://github.com/username/comed/raw/main/docs/images/workflow.png" alt="CoMed Workflow" width="700"/>
</p>
## ✨ Features
| Feature | Description |
|---------|-------------|
| 🔍 **Literature Mining** | Automatically search PubMed for scientific papers discussing specific drug combinations |
| 🧠 **Chain-of-Thought Analysis** | Use LLM-based reasoning to identify true drug combination mentions and extract relevant information |
| ⚠️ **Multi-dimensional Risk Analysis** | Evaluate risks across five key clinical dimensions: side effects, efficacy & safety, indications, patient selection, and management |
| 📊 **Evidence Synthesis** | Aggregate and synthesize findings from multiple sources with proper citation |
| 📈 **Scanpy-inspired API** | Intuitive, chainable API inspired by Scanpy making analysis workflows easy to build |
| 📱 **Interactive Reports** | Generate rich HTML reports with detailed findings and direct links to source papers |
| 🔗 **Research Traceability** | All findings include references to original medical literature for verification |
| 🔄 **Model Agnostic** | Compatible with multiple LLM providers (OpenAI, Anthropic, QianWen, etc.) |
## 📦 Installation
### Prerequisites
- Python 3.7 or later
- pip package manager
- Access to an LLM API (OpenAI, Anthropic, etc.)
### Quick Installation
```bash
# Install the stable release
pip install comed
# Or install with visualization dependencies
pip install comed[vis]
# For development version with latest features
pip install git+https://github.com/username/comed.git
```
### Dependencies
CoMed has the following dependencies, which are automatically installed:
```
pandas>=1.3.0
numpy>=1.20.0
biopython>=1.79
tqdm>=4.62.0
openai>=1.0.0
requests>=2.26.0
matplotlib>=3.4.0  # Optional for visualization
seaborn>=0.11.0    # Optional for visualization
```
### Manual Installation from Source
```bash
git clone https://github.com/username/comed.git
cd comed
pip install -e .
```
## 🚀 Quick Start
### Basic Usage
```python
import os
import comed
# Configure your LLM API credentials
os.environ["MODEL_NAME"] = "gpt-4" 
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key"
# Initialize CoMed with a list of drugs
drugs = ["warfarin", "aspirin", "ibuprofen", "clopidogrel"]
com = comed.CoMedData(drugs)
# Run the full analysis pipeline
report_path = com.run_full_analysis(retmax=30)
print(f"Report generated at: {report_path}")
```
### Expected Output
```
INFO:comed:Initializing CoMed with 4 drugs and 6 drug combinations
INFO:comed:Using model: gpt-4
INFO:comed:Searching PubMed for drug combination: warfarin + aspirin [1/6]
INFO:comed:Found 30 papers for warfarin + aspirin
INFO:comed:Searching PubMed for drug combination: warfarin + ibuprofen [2/6]
INFO:comed:Found 12 papers for warfarin + ibuprofen
...
INFO:comed:Analyzing associations for 102 papers
INFO:comed:Found 47 papers with confirmed drug combinations
INFO:comed:Analyzing risks across 5 dimensions
INFO:comed:Risk analysis complete - processed 47 papers
INFO:comed:Generating HTML report
INFO:comed:Report saved to: CoMed_Risk_Analysis_Report.html
```
### Step-by-Step Analysis
For more control, you can run each step of the pipeline separately:
```python
import comed
import logging
# Enable detailed logging
comed.utils.configure_logging(log_level=logging.INFO)
# Initialize with drug list
drugs = ["metformin", "lisinopril", "atorvastatin"]
com = comed.CoMedData(drugs)
# Step 1: Search medical literature
com.search(retmax=30, email="your.email@example.com")
print(f"Found {len(com.papers)} papers")
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
CoMed is model-agnostic and can work with any compatible LLM API:
```python
import comed
# Initialize with drugs
com = comed.CoMedData(["simvastatin", "amlodipine"])
# Configure for specific LLM
com.set_config({
    'model_name': 'qwen2.5-32b-instruct',
    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'your-dashscope-api-key'
})
# Run analysis
com.run_full_analysis(retmax=20)
```
### LLM Compatibility Table
| LLM Provider | Supported Models | Performance | Setup Example |
|--------------|------------------|-------------|---------------|
| 🟢 **OpenAI** | GPT-4, GPT-3.5-Turbo | Excellent | `model_name='gpt-4', api_base='https://api.openai.com/v1'` |
| 🟢 **Anthropic** | Claude 3 Opus, Claude 3 Sonnet | Excellent | `model_name='claude-3-opus-20240229', api_base='https://api.anthropic.com/v1'` |
| 🟢 **QianWen** | Qwen2.5, Qwen-Max | Very Good | `model_name='qwen2.5-32b-instruct', api_base='https://dashscope.aliyuncs.com/compatible-mode/v1'` |
| 🟡 **Mistral AI** | Mistral Large, Medium | Good | `model_name='mistral-large-latest', api_base='https://api.mistral.ai/v1'` |
| 🟡 **Cohere** | Command R+, Command R | Good | `model_name='command-r-plus', api_base='https://api.cohere.com/v1'` |
| 🟡 **Llama 3** | Llama 3 70B, 8B | Moderate | `model_name='llama-3-70b-instruct', api_base='your-llama-endpoint'` |
## 🔍 Understanding the Pipeline
CoMed's pipeline consists of four main stages:
<table>
  <tr>
    <th>Stage</th>
    <th>Description</th>
    <th>Key Features</th>
  </tr>
  <tr>
    <td>📚 <b>Literature Search</b></td>
    <td>Searches PubMed for papers mentioning both drugs in each combination</td>
    <td>
      • Customizable search parameters<br>
      • Error handling and retry logic<br>
      • Filters papers without abstracts
    </td>
  </tr>
  <tr>
    <td>🧩 <b>Association Analysis</b></td>
    <td>Uses CoT reasoning to determine if papers genuinely discuss drug combinations</td>
    <td>
      • Four-step reasoning process<br>
      • Identifies false positives<br>
      • Provides justification for each decision
    </td>
  </tr>
  <tr>
    <td>⚖️ <b>Risk Evaluation</b></td>
    <td>Analyzes five key dimensions of drug interaction risk</td>
    <td>
      • Side effects & adverse events<br>
      • Efficacy & safety<br>
      • Indications & contraindications<br>
      • Patient selection criteria<br>
      • Monitoring & management recommendations
    </td>
  </tr>
  <tr>
    <td>📊 <b>Report Generation</b></td>
    <td>Synthesizes findings into comprehensive HTML reports</td>
    <td>
      • Overall risk assessment<br>
      • Dimension-specific summaries<br>
      • Citation links to original papers<br>
      • Interactive format
    </td>
  </tr>
</table>
### Pipeline Visualization
```
Drugs → Combinations → Literature Search → Filter Papers → CoT Analysis → 
Risk Dimension Analysis → Evidence Synthesis → HTML Report
```
## 📘 API Reference
### CoMedData Class
The core class that handles the entire analysis pipeline.
```python
class CoMedData:
    """
    Main class for drug co-medication analysis.
    Attributes:
        drugs (list): List of drug names to analyze
        drug_combinations (list): Generated pairs of drugs to analyze
        papers (DataFrame): Papers retrieved from literature search
        associations (DataFrame): Results of association analysis
        risk_analysis (DataFrame): Results of risk dimension analysis
    """
    def __init__(self, drugs=None):
        """Initialize CoMedData with optional list of drugs."""
    def set_config(self, config) -> CoMedData:
        """Set configuration parameters for the analysis."""
    def add_drugs(self, drugs) -> CoMedData:
        """Add more drugs to the analysis."""
    def search(self, retmax=30, email='your_email@example.com', retry=3, 
               delay=3, filepath="ddc_papers.csv", verbose=True) -> CoMedData:
        """Search PubMed for papers on drug combinations."""
    def analyze_associations(self, filepath="ddc_papers_association_pd.csv", 
                            verbose=True, max_retries=30, retry_delay=5) -> CoMedData:
        """Analyze if papers discuss actual drug combinations."""
    def analyze_risks(self, filepath="ddc_papers_risk.csv", verbose=True) -> CoMedData:
        """Analyze different dimensions of risk."""
    def generate_report(self, output_file="CoMed_Risk_Analysis_Report.html", 
                       verbose=True) -> str:
        """Generate HTML report of findings."""
    def run_full_analysis(self, retmax=30, verbose=True) -> str:
        """Run the complete analysis pipeline."""
```
### IO Functions
Functions to load and save analysis data.
```python
import comed.io as io
# Load data
papers = io.read_papers(filepath)
associations = io.read_associations(filepath)
risk_analysis = io.read_risk_analysis(filepath)
# Save data
io.save_dataframe(df, filepath, index=False)
```
### Utility Functions
```python
import comed.utils as utils
# Configure logging
utils.configure_logging(log_file="comed_analysis.log", log_level=logging.DEBUG)
# Text processing utilities
clean_text = utils.clean_text(text)
json_data = utils.extract_json(text)
```
## ⚙️ Advanced Configuration
### Environment Variables
You can set these environment variables before running CoMed:
```python
import os
# LLM Configuration
os.environ["MODEL_NAME"] = "gpt-4"  # or any compatible model
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key"
os.environ["LOG_DIR"] = "logs"  # Directory for log files
os.environ["OLD_OPENAI_API"] = "No"  # Use "Yes" for older OpenAI API versions
```
### Advanced Parameter Configuration
```python
import comed
com = comed.CoMedData(drugs)
com.set_config({
    # LLM parameters
    'model_name': 'gpt-4',
    'api_base': 'https://api.openai.com/v1',
    'api_key': 'your-api-key',
    'temperature': 0.0,  # Lower temperature for more deterministic outputs
    'max_tokens': 4000,  # Increase token limit for longer responses
    # Analysis parameters
    'reasoning_steps': 4,  # Number of CoT reasoning steps
    'retry_count': 5,      # Number of retries for API calls
    'delay': 2,            # Delay between API calls
    # Logging
    'verbose': True,       # Detailed console output
    'log_level': 'INFO',   # Logging level
})
```
### Customizing Chain-of-Thought Reasoning
For advanced users, you can customize the reasoning steps used in analysis:
```python
import comed
# Create a custom reasoning template
custom_cot = [
    "Question 1: Identify specific sentences that mention both {drug1} and {drug2} together.",
    "Question 2: Evaluate if these drugs are mentioned as being administered together or separately.",
    "Question 3: Is there explicit discussion of their interaction or combined effects?",
    "Question 4: Based on the evidence, determine if this paper genuinely discusses co-administration."
]
# Apply the custom template
com = comed.CoMedData(drugs)
com.set_config({
    'cot_template': custom_cot,
    'cot_format': 'json'  # Force structured JSON output
})
```
## 🔬 Real-World Examples
### Case 1: Analyzing Anticoagulant Interactions
```python
import comed
import pandas as pd
import matplotlib.pyplot as plt
# Focus on anticoagulants and common co-medications
drugs = ["warfarin", "apixaban", "rivaroxaban", "clopidogrel", "aspirin"]
com = comed.CoMedData(drugs)
# Detailed step-by-step approach
com.search(retmax=50)
# Filter to recent papers
recent_papers = com.papers[com.papers["Publication Date"].str.contains("202")]
com.papers = recent_papers
# Analyze papers
com.analyze_associations()
com.analyze_risks()
# Generate report
report_path = com.generate_report("anticoagulant_interactions.html")
# Visual analysis of confirmed interactions
confirmed = com.associations[com.associations["Combined_medication"].str.lower() == "yes"]
pairs = confirmed.groupby(["Drug1", "Drug2"]).size().reset_index(name="count")
plt.figure(figsize=(10, 6))
plt.bar(pairs.apply(lambda x: f"{x['Drug1']} + {x['Drug2']}", axis=1), pairs["count"])
plt.xticks(rotation=45, ha="right")
plt.title("Confirmed Drug Combinations in Literature")
plt.tight_layout()
plt.savefig("drug_combinations.png")
```
### Case 2: Analyzing Diabetes Medication Interactions with Statistical Analysis
```python
import comed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Common diabetes medications and statins
diabetes_meds = ["metformin", "insulin", "sitagliptin", "liraglutide"]
statins = ["atorvastatin", "simvastatin", "rosuvastatin"]
com = comed.CoMedData(diabetes_meds + statins)
# Run full analysis
com.run_full_analysis(retmax=40)
# Extract data for further analysis
risk_data = com.risk_analysis.copy()
# Create a matrix of interactions
matrix_data = []
for drug1 in diabetes_meds:
    for drug2 in statins:
        risk_count = len(risk_data[
            ((risk_data["Drug1"] == drug1) & (risk_data["Drug2"] == drug2) |
             (risk_data["Drug1"] == drug2) & (risk_data["Drug2"] == drug1)) &
            (risk_data["Risks"] != "Invalid")
        ])
        matrix_data.append({"Diabetes Med": drug1, "Statin": drug2, "Risk Count": risk_count})
# Create and visualize interaction matrix
matrix_df = pd.DataFrame(matrix_data)
matrix_pivot = matrix_df.pivot(index="Diabetes Med", columns="Statin", values="Risk Count")
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_pivot, annot=True, cmap="YlOrRd", fmt="d")
plt.title("Number of Papers Reporting Risks for Drug Combinations")
plt.tight_layout()
plt.savefig("interaction_matrix.png")
```
### Case 3: Batch Analysis with Clinical Focus
```python
import comed
import pandas as pd
import os
# Define drug classes for comprehensive analysis
antihypertensives = ["lisinopril", "amlodipine", "hydrochlorothiazide", "losartan"]
antidiabetics = ["metformin", "sitagliptin", "insulin", "empagliflozin"]
statins = ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"]
anticoagulants = ["warfarin", "apixaban", "rivaroxaban", "dabigatran"]
# Create output directory
os.makedirs("interaction_reports", exist_ok=True)
# Function to analyze interactions between two drug classes
def analyze_class_interactions(class1, class2, class1_name, class2_name):
    print(f"Analyzing {class1_name} vs {class2_name}...")
    com = comed.CoMedData(class1 + class2)
    com.search(retmax=20)
    com.analyze_associations()
    com.analyze_risks()
    report_path = com.generate_report(f"interaction_reports/{class1_name}_{class2_name}_interactions.html")
    return com
# Run analyses between all class pairs
results = []
classes = [
    (antihypertensives, "antihypertensives"),
    (antidiabetics, "antidiabetics"),
    (statins, "statins"),
    (anticoagulants, "anticoagulants")
]
for i, (class1, class1_name) in enumerate(classes):
    for j, (class2, class2_name) in enumerate(classes[i+1:], i+1):
        com = analyze_class_interactions(class1, class2, class1_name, class2_name)
        # Collect statistics
        positive_count = len(com.associations[com.associations['Combined_medication'].str.lower() == 'yes'])
        risk_count = len(com.risk_analysis[com.risk_analysis['Risks'] != 'Invalid'])
        results.append({
            "Class 1": class1_name,
            "Class 2": class2_name,
            "Total Papers": len(com.papers),
            "Confirmed Combinations": positive_count,
            "Risk Mentions": risk_count
        })
# Create summary report
summary_df = pd.DataFrame(results)
summary_df.to_csv("interaction_reports/class_interaction_summary.csv", index=False)
print("Analysis complete. Summary saved to 'interaction_reports/class_interaction_summary.csv'")
```
## 📊 Performance Benchmarks
### System Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.7+ | 3.9+ |
| RAM | 4 GB | 8+ GB |
| CPU | Dual Core | Quad Core |
| Disk Space | 500 MB | 2+ GB |
| Internet | Required | Required (Stable) |
### Performance by Drug Count
| Drug Count | Combinations | Approx. Runtime* | API Calls | Estimated Cost** |
|------------|--------------|------------------|-----------|------------------|
| 3 drugs | 3 pairs | 5-10 min | ~50 | $1-3 |
| 5 drugs | 10 pairs | 20-30 min | ~150 | $3-9 |
| 10 drugs | 45 pairs | 1.5-2.5 hours | ~700 | $15-45 |
| 15 drugs | 105 pairs | 3.5-5 hours | ~1600 | $35-105 |
\* _Using GPT-4 with retmax=30, varies by LLM and paper count_
\** _Estimated cost using OpenAI GPT-4 API, varies by provider_
### LLM Performance Comparison
| Model | Accuracy | Speed | Cost Efficiency | Overall |
|-------|----------|-------|-----------------|---------|
| GPT-4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Claude 3 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| GPT-3.5 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Qwen2.5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Llama 3 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Mistral | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
## 🤝 Contributing
Contributions are welcome and greatly appreciated! CoMed is an academic open-source project, and we value all forms of contribution.
### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 🧪 Add test cases
- 📚 Improve documentation
- 🔄 Submit pull requests
### Contribution Process
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add appropriate tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request
Please see our [CONTRIBUTING.md](https://github.com/username/comed/blob/main/CONTRIBUTING.md) file for detailed guidelines.
## 📜 License
This project is licensed under the BSD 2-Clause License - see the [LICENSE](https://github.com/username/comed/blob/main/LICENSE) file for details.
```
BSD 2-Clause License
Copyright (c) 2023, CoMed Developers
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
## 📝 Citation
If you use CoMed in your research, please cite:
```bibtex
@software{comed2023,
  author = {Your Name and Contributors},
  title = {CoMed: A Framework for Analyzing Co-Medication Risks using Chain-of-Thought Reasoning},
  year = {2023},
  url = {https://github.com/username/comed},
  version = {0.1.0},
  doi = {10.5281/zenodo.12345678}
}
```
For academic publications, you can also cite our paper:
```bibtex
@article{comed2023paper,
  title={Automated Analysis of Drug Co-Medication Risks using Chain-of-Thought Reasoning and Large Language Models},
  author={Your Name and Collaborators},
  journal={Journal of Biomedical Informatics},
  year={2023},
  volume={},
  pages={},
  publisher={Elsevier},
  doi={10.xxxx/xxxx.xxxx.xxxx}
}
```
## 📱 Contact
- **Project Lead**: Your Name - email@example.com
- **Website**: https://comed.readthedocs.io
- **GitHub Issues**: https://github.com/username/comed/issues
- **Twitter**: [@CoMedProject](https://twitter.com/CoMedProject)
## ❓ FAQ
<details>
<summary><b>Q: How many drugs can I analyze at once?</b></summary>
**A:** CoMed can analyze any number of drugs, but the number of combinations grows quadratically (n*(n-1)/2). For optimal performance:
- 5-10 drugs is ideal for quick analyses
- 10-20 drugs is suitable for comprehensive reviews
- 20+ drugs may require batch processing to manage computational resources
You can use `add_drugs()` to incrementally add drugs to your analysis.
</details>
<details>
<summary><b>Q: Which LLMs work best with CoMed?</b></summary>
**A:** CoMed performs best with models that excel at chain-of-thought reasoning:
- **Best Performance**: GPT-4, Claude 3 Opus
- **Good Balance**: GPT-3.5-Turbo, Claude 3 Sonnet, Qwen-Max
- **More Affordable**: Mistral, Qwen2.5, Llama 3
More capable models produce higher quality medical analyses, particularly for complex pharmacological interactions.
</details>
<details>
<summary><b>Q: How can I interpret the risk levels in the report?</b></summary>
**A:** CoMed summarizes findings from the literature without assigning definitive risk levels. The reports provide:
1. Evidence from medical papers with context
2. Citations to original sources
3. Multi-dimensional analysis of risks
These findings should be evaluated by healthcare professionals in the context of specific patients and clinical scenarios.
</details>
<details>
<summary><b>Q: How recent are the papers CoMed analyzes?</b></summary>
**A:** CoMed searches PubMed for the most relevant papers based on your query terms. By default, results are sorted by relevance. You can:
- Filter papers by date in post-processing
- Modify the search query to focus on more recent literature
- Use the PubMed "sort" parameter to prioritize recent publications
</details>
<details>
<summary><b>Q: Is CoMed suitable for clinical decision-making?</b></summary>
**A:** CoMed is designed as a research tool and information aggregator. It should be used to:
- Generate hypotheses for further research
- Provide literature-based evidence summaries
- Support, not replace, clinical judgment
Any information used for clinical decisions should be verified by qualified healthcare professionals and cross-checked with authoritative drug interaction resources.
</details>
<details>
<summary><b>Q: Can I use CoMed without an internet connection?</b></summary>
**A:** CoMed requires internet access for two key functions:
1. Searching PubMed for medical literature
2. Accessing LLM APIs for analysis
If you have pre-downloaded papers and abstracts, you could potentially modify CoMed to use local LLM inference, but this is not supported in the standard package.
</details>
<details>
<summary><b>Q: How is CoMed different from traditional drug interaction databases?</b></summary>
**A:** Unlike static drug interaction databases, CoMed:
- Dynamically searches the latest scientific literature
- Provides direct links to evidence
- Evaluates multiple risk dimensions
- Includes the reasoning process
- Can analyze novel or uncommon drug combinations not yet in databases
- Generates comprehensive reports with nuanced analysis
</details>
<details>
<summary><b>Q: Can I customize the risk dimensions that CoMed analyzes?</b></summary>
**A:** Yes, advanced users can customize risk dimensions by modifying the risk analysis templates in the codebase. This requires some Python knowledge but allows you to focus on specific aspects relevant to your research.
</details>
## 📈 Roadmap
- **0.2.0**: Advanced filtering options and improved document processing
- **0.3.0**: Integration with DrugBank and other pharmacological databases
- **0.4.0**: Interactive visualization tools and dashboards
- **0.5.0**: Support for multilingual paper analysis
- **1.0.0**: Comprehensive pharmacological reasoning with full documentation and expanded test suite
