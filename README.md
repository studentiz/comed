# CoMed: Drug Co-Medication Risk Analysis Framework
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
CoMed is a comprehensive framework for analyzing drug co-medication risks using Chain-of-Thought (CoT) reasoning and large language models. It automates the process of searching medical literature, analyzing drug interactions, and generating detailed risk assessment reports.
![CoMed Workflow](https://github.com/username/comed/raw/main/docs/images/workflow.png)
## Features
- 🔍 **Literature Search**: Automatically search PubMed for papers discussing specific drug combinations
- 🧠 **Chain-of-Thought Analysis**: Use LLM-based reasoning to identify true drug combination mentions
- ⚠️ **Risk Assessment**: Evaluate multiple dimensions of drug interaction risks
- 📊 **Comprehensive Reporting**: Generate interactive HTML reports with rich information
- 📈 **Scanpy-like API**: Intuitive, chainable API inspired by Scanpy making analysis workflows easy to build
## Installation
### Quick Installation
```bash
pip install comed
```
### Install from GitHub (Latest Development Version)
```bash
pip install git+https://github.com/username/comed.git
```
### Requirements
CoMed requires Python 3.7+ and the following dependencies:
- pandas
- numpy
- biopython
- tqdm
- openai (v1.0.0+)
- requests
## Quick Start
### Basic Usage
```python
import os
import comed
# Configure your LLM API credentials
os.environ["MODEL_NAME"] = "gpt-4" # or any compatible model
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key"
# Initialize CoMed with a list of drugs
drugs = ["warfarin", "aspirin", "ibuprofen", "clopidogrel"]
com = comed.CoMedData(drugs)
# Run the full analysis pipeline
report_path = com.run_full_analysis(retmax=30)
print(f"Report generated at: {report_path}")
```
### Step-by-Step Analysis
For more control, you can run each step of the pipeline separately:
```python
import comed
# Initialize with drug list
drugs = ["metformin", "lisinopril", "atorvastatin"]
com = comed.CoMedData(drugs)
# Step 1: Search medical literature
# This will find papers that mention each drug combination
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
### Working with Existing Data
You can load previously saved data to continue analysis:
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
You can add more drugs to an existing analysis:
```python
import comed
# Start with a few drugs
com = comed.CoMedData(["warfarin", "aspirin"])
com.search(retmax=30)
# Add more drugs later
com.add_drugs(["heparin", "clopidogrel"])
com.search(retmax=30)  # Only searches the new combinations
# Complete the analysis
com.analyze_associations() \
    .analyze_risks() \
    .generate_report("anticoagulant_interactions.html")
```
## Understanding the Output
CoMed generates a comprehensive HTML report with multiple sections:
![CoMed Report](https://github.com/username/comed/raw/main/docs/images/report.png)
1. **Metadata Section**: Contains information about the analysis, including:
   - When the report was created
   - Which LLM model was used
   - Number of drug combinations analyzed
   - Number of papers evaluated
2. **Drug Combination Sections**: Each drug pair has its own section with:
   - Overall risk assessment
   - Detailed analysis of side effects
   - Efficacy and safety information
   - Indications and patient selection criteria
   - Clinical management recommendations
   - References to the source papers
3. **References**: Each finding includes numbered references that link to the original PubMed papers.
## API Reference
### CoMedData Class
The core class that handles the entire analysis pipeline.
```python
class CoMedData:
    def __init__(self, drugs=None)
    def set_config(self, config) -> CoMedData
    def add_drugs(self, drugs) -> CoMedData
    def search(self, retmax=30, email='your_email@example.com', retry=3, 
               delay=3, filepath="ddc_papers.csv", verbose=True) -> CoMedData
    def analyze_associations(self, filepath="ddc_papers_association_pd.csv", 
                            verbose=True, max_retries=30, retry_delay=5) -> CoMedData
    def analyze_risks(self, filepath="ddc_papers_risk.csv", verbose=True) -> CoMedData
    def generate_report(self, output_file="CoMed_Risk_Analysis_Report.html", 
                       verbose=True) -> str
    def run_full_analysis(self, retmax=30, verbose=True) -> str
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
## Advanced Configuration
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
### Logging Configuration
CoMed uses Python's standard logging module. You can configure logging behavior:
```python
import comed.utils as utils
utils.configure_logging(log_file="comed_analysis.log", log_level=logging.DEBUG)
```
## Example Use Cases
### Case 1: Analyzing Common Drug Combinations in Elderly Patients
```python
import comed
# Common medications prescribed to elderly patients
drugs = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "levothyroxine"]
com = comed.CoMedData(drugs)
# Run analysis with detailed console output
report = com.run_full_analysis(retmax=50)
```
### Case 2: Focused Analysis on Anticoagulant Interactions
```python
import comed
# Focus on anticoagulants and common co-medications
drugs = ["warfarin", "apixaban", "rivaroxaban", "clopidogrel", "aspirin"]
com = comed.CoMedData(drugs)
# Detailed step-by-step approach
com.search(retmax=50)
# Filter papers by publication date (if needed)
recent_papers = com.papers[com.papers["Publication Date"].str.contains("202")]
com.papers = recent_papers
# Continue analysis with filtered papers
com.analyze_associations() \
   .analyze_risks() \
   .generate_report("anticoagulant_interactions.html")
```
### Case 3: Batch Analysis of Multiple Drug Classes
```python
import comed
# Analyze interactions between drug classes sequentially
antihypertensives = ["lisinopril", "amlodipine", "hydrochlorothiazide"]
antidiabetics = ["metformin", "sitagliptin", "insulin"]
statins = ["atorvastatin", "simvastatin", "rosuvastatin"]
# First analyze antihypertensives with antidiabetics
com = comed.CoMedData(antihypertensives + antidiabetics)
com.run_full_analysis(retmax=20)
# Then analyze antidiabetics with statins
com2 = comed.CoMedData(antidiabetics + statins)
com2.run_full_analysis(retmax=20)
```
## How It Works
CoMed uses a four-step process to analyze co-medication risks:
1. **Literature Search**: For each drug pair, CoMed queries PubMed to find relevant medical papers that mention both drugs.
2. **Association Analysis**: Using Chain-of-Thought reasoning, the LLM analyzes each paper to determine if it genuinely discusses the combined use of the drugs (not just mentioning them separately).
3. **Risk Evaluation**: For papers with confirmed drug combinations, CoMed analyzes five key dimensions:
   - Risks & side effects
   - Efficacy & safety
   - Indications & contraindications
   - Patient population & selection
   - Monitoring & management
4. **Report Generation**: CoMed summarizes the findings into a comprehensive HTML report, providing clinically relevant information with references to the source literature.
## Customization
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
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Citation
If you use CoMed in your research, please cite:
```
@software{comed2023,
  author = {Your Name},
  title = {CoMed: A Framework for Analyzing Co-Medication Risks using Chain-of-Thought Reasoning},
  year = {2023},
  url = {https://github.com/username/comed},
}
```
## Contact
For questions or support, please open an issue on the GitHub repository or contact the maintainer at studentiz@live.com.
---
## FAQ
### Q: How many drugs can I analyze at once?
**A:** CoMed can analyze any number of drugs, but the number of combinations grows quickly (n*(n-1)/2). For larger analyses, consider breaking it into smaller drug groups to manage computational resources.
### Q: Which LLMs work best with CoMed?
**A:** CoMed performs best with more advanced models like GPT-4, Claude 3 Opus, or Qwen Max, but will work with any model that can handle chain-of-thought reasoning. More capable models produce higher quality medical analyses.
### Q: How can I interpret the risk levels in the report?
**A:** CoMed summarizes findings from the literature but does not assign specific risk levels. The reports include the context and evidence from medical papers, which should be evaluated by healthcare professionals.
### Q: How recent are the papers CoMed analyzes?
**A:** CoMed searches PubMed for the most relevant papers based on your query terms. You can filter papers by date in post-processing if you need more recent literature.
### Q: Is CoMed suitable for clinical decision-making?
**A:** CoMed is designed as a research tool and information aggregator. Any information used for clinical decisions should be verified by qualified healthcare professionals.
