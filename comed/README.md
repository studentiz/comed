# CoMed
CoMed is a Python package for analyzing co-medication risks using Chain-of-Thought (CoT) reasoning with large language models. It provides a comprehensive framework for retrieving scientific literature, identifying drug combinations, analyzing potential risks, and generating detailed reports.
## Features
- 🔍 **Automated Literature Search**: Retrieve relevant papers from PubMed for drug combinations
- 🧠 **Chain-of-Thought Analysis**: Use step-by-step reasoning to analyze drug associations in scientific abstracts
- ⚠️ **Multi-dimensional Risk Assessment**: Evaluate risks, safety, indications, patient selectivity, and management strategies
- 📊 **Interactive HTML Reports**: Generate comprehensive reports with reference tracking
## Installation
Install from PyPI:
```bash
pip install comed
```
## Dependencies
CoMed requires the following dependencies:
- pandas (>=2.2.2)
- numpy (>=1.26.0)
- biopython (>=1.85)
- tqdm (>=4.67.1)
- openai (>=1.65.1)
- requests (>=2.32.3)
## Quick Start
```python
import os
import comed
# Set required environment variables
os.environ["MODEL_NAME"] = "gpt-4o"  # or any other supported model
os.environ["API_BASE"] = "https://api.openai.com/v1"  # default OpenAI endpoint
os.environ["API_KEY"] = "your-api-key-here"
os.environ["LOG_DIR"] = "logs"  # directory for log files
# Create a CoMed instance with list of drugs
drugs = ["warfarin", "aspirin", "heparin", "clopidogrel"]
com = comed.CoMedData(drugs)
# Run the full analysis pipeline
report_path = com.run_full_analysis(retmax=30)
print(f"Report generated at: {report_path}")
```
## Usage Examples
### Example 1: Step-by-Step Analysis
Run each step of the analysis separately for more control:
```python
import os
import comed
# Set API credentials
os.environ["MODEL_NAME"] = "gpt-4o"
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key-here"
# Create a CoMed instance
drugs = ["warfarin", "aspirin", "penicillin"]
com = comed.CoMedData(drugs)
# Search PubMed for literature (up to 20 papers per combination)
com.search(retmax=20, email="your.email@example.com")
# Analyze which papers mention drug combinations
com.analyze_associations()
# Evaluate risks across multiple dimensions
com.analyze_risks()
# Generate an HTML report
com.generate_report("Anticoagulant_Report.html")
```
### Example 2: Using Method Chaining
CoMed supports a Scanpy-style API with method chaining:
```python
import os
import comed
# Set API credentials
os.environ["MODEL_NAME"] = "gpt-4o"
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key-here"
# Create a CoMed instance and run analysis pipeline with method chaining
drugs = ["ibuprofen", "naproxen", "acetaminophen"]
com = comed.CoMedData(drugs)
com.search(retmax=30) \
   .analyze_associations() \
   .analyze_risks() \
   .generate_report("NSAID_Interactions.html")
```
### Example 3: Adding Drugs Incrementally
You can add more drugs to your analysis after initial setup:
```python
import os
import comed
# Set API credentials
os.environ["MODEL_NAME"] = "gpt-4o"
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key-here"
# Start with a smaller set of drugs
com = comed.CoMedData(["warfarin", "aspirin"])
com.search(retmax=30)
# Add more drugs later
com.add_drugs(["heparin", "clopidogrel"])
# Only search for the new combinations
com.search(retmax=30)
# Complete the analysis pipeline
com.analyze_associations() \
   .analyze_risks() \
   .generate_report("Expanded_Drug_Report.html")
```
### Example 4: Custom Configuration
Set custom configuration options:
```python
import comed
# Create CoMed instance
com = comed.CoMedData(["metformin", "glyburide", "insulin"])
# Set custom configuration
com.set_config({
    'model_name': 'gpt-4o',
    'api_key': 'your-api-key-here',
    'api_base': 'https://api.openai.com/v1',
    'old_openai_api': 'No'
})
# Run analysis
com.run_full_analysis(retmax=20)
```
### Example 5: Working with Existing Data
Load previously saved data to continue analysis:
```python
import comed
import comed.io as io
# Load data from previous runs
papers = io.read_papers("saved_papers.csv")
associations = io.read_associations("saved_associations.csv")
# Create CoMed instance with pre-loaded data
com = comed.CoMedData()
com.papers = papers
com.associations = associations
# Continue analysis from where you left off
com.analyze_risks().generate_report("Updated_Report.html")
```
## API Reference
### Core Class
`CoMedData`: The main class for managing drug combinations and analysis
Constructor parameters:
- `drugs` (optional): List of drug names to analyze
Main methods:
- `search()`: Search PubMed for papers on drug combinations
- `analyze_associations()`: Analyze which papers mention drug combinations
- `analyze_risks()`: Perform risk analysis on confirmed drug combinations
- `generate_report()`: Create an HTML report of findings
- `run_full_analysis()`: Run the complete analysis pipeline
### Environment Variables
- `MODEL_NAME`: Name of the LLM to use (e.g., "gpt-4", "qwen2.5-32b-instruct")
- `API_BASE`: Base URL for the LLM API
- `API_KEY`: API key for LLM access
- `LOG_DIR`: Directory to store log files
- `OLD_OPENAI_API`: Whether to use the old OpenAI API format ("Yes" or "No")
## Development
Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request