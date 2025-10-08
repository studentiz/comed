# CoMed: A Framework for Drug Co-Medication Risk Analysis

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI version](https://badge.fury.io/py/comed.svg?icon=si%3Apython)](https://badge.fury.io/py/comed)

## 🎯 Overview

CoMed is a comprehensive framework for analyzing drug co-medication risks using a modular architecture that supports RAG (Retrieval-Augmented Generation), CoT (Chain-of-Thought reasoning), and Multi-Agent systems. This version addresses reviewer feedback by providing clear component separation, ablation study support, and true multi-agent collaboration.

## 🔧 Key Features

### 1. Modular Architecture
- **RAG Module** (`rag.py`): Independent literature retrieval system
- **CoT Module** (`cot.py`): Chain-of-thought reasoning system  
- **Multi-Agent Module** (`agents.py`): True agent-to-agent collaboration
- **Benchmark Module** (`benchmark.py`): Ablation studies and performance evaluation

### 2. Ablation Study Support
- Independent testing of each component's contribution
- Detailed performance comparison and analysis
- Automatic generation of ablation study reports

### 3. True Multi-Agent System
- Agent-to-agent communication and collaboration
- Specialized agents for different roles
- Agent state management and conversation tracking

## 📦 Installation

```bash
pip install comed
```

## 🚀 Quick Start

### Basic Usage

```python
import os
import comed

# Set required environment variables
os.environ["MODEL_NAME"] = "gpt-4o"
os.environ["API_BASE"] = "https://api.openai.com/v1"
os.environ["API_KEY"] = "your-api-key-here"

# Initialize system
drugs = ["warfarin", "aspirin", "ibuprofen"]
com = comed.CoMedData(drugs)

# Run full analysis
report_path = com.run_full_analysis(retmax=30, verbose=True)
print(f"Report generated at: {report_path}")
```

## 📚 Usage Examples

### Example 1: Step-by-Step Analysis

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

# Search PubMed for literature
com.search(retmax=20, email="your.email@example.com")

# Analyze which papers mention drug combinations
com.analyze_associations()

# Evaluate risks across multiple dimensions
com.analyze_risks()

# Generate an HTML report
com.generate_report("Anticoagulant_Report.html")
```

### Example 2: Method Chaining

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

### Example 4: Ablation Study

```python
import comed

# Initialize system
drugs = ["metformin", "lisinopril", "atorvastatin"]
com = comed.CoMedData(drugs)

# Run ablation study
ablation_results = com.run_ablation_study(retmax=20, verbose=True)

# View results
print("Ablation Study Results:")
for stage, result in ablation_results["ablation_results"].items():
    if stage != "ablation_report":
        print(f"{stage}: {result['time']:.1f}s, Papers: {result['stats']['total_papers']}")

# Component comparison
comparison_results = com.compare_components(retmax=20, verbose=True)
report = comparison_results["comparison_report"]

print("\nPerformance Comparison:")
for component, perf in report["performance_summary"].items():
    print(f"{component}: {perf['time']:.1f}s, Efficiency: {perf['efficiency']:.2f}")
```

### Example 5: Multi-Agent System

```python
from comed import MultiAgentSystem

# Initialize multi-agent system
agent_system = MultiAgentSystem(
    model_name="gpt-4o",
    api_key="your-key",
    api_base="https://api.openai.com/v1"
)

# Process drug combination
drug1, drug2 = "warfarin", "aspirin"
abstract = "Literature abstract content..."

# Multi-agent collaboration analysis
result = agent_system.process_drug_combination(drug1, drug2, abstract)

print("Multi-Agent Analysis Results:")
print(f"Risk Analysis: {result['risk_analysis']}")
print(f"Safety Assessment: {result['safety_assessment']}")
print(f"Clinical Recommendation: {result['clinical_recommendation']}")
```

## 🎮 Demo Examples

Run the comprehensive demo to see CoMed in action:

```bash
# Run basic demo
python examples/basic_demo.py

# Run ablation study demo
python examples/ablation_study_example.py

# Run quick start demo
python examples/quick_start.py
```

### Using Existing Data

If you already have association data (e.g., `ddc_papers_association_pd.csv`), you can skip the search and analysis steps:

```bash
# Load existing data and run multi-agent analysis
python examples/load_and_analyze.py

# Simple multi-agent test
python examples/simple_agent_test.py

# Direct multi-agent test
python examples/direct_agent_test.py
```

The demo includes:
- Basic usage examples
- Step-by-step analysis
- Method chaining
- Incremental drug addition
- Data persistence
- Ablation studies
- Multi-agent collaboration
- Direct multi-agent testing with existing data

## 🔧 Advanced Configuration

### Environment Variables

```bash
export MODEL_NAME="gpt-4o"
export API_BASE="https://api.openai.com/v1"
export API_KEY="your-api-key"
export LOG_DIR="logs"
```

### Custom Configuration

```python
# Configure LLM
com = comed.CoMedData(["warfarin", "aspirin"])
com.set_config({
    'model_name': 'gpt-4o',
    'api_base': 'https://api.openai.com/v1',
    'api_key': 'your-key'
})
```

## 📈 Performance Evaluation

### Ablation Study Metrics

- **Time Efficiency**: Processing time for each component
- **Quality Metrics**: Positive association rate, accuracy
- **Component Contributions**: Independent contribution of each component
- **Efficiency Analysis**: Balance between processing speed and quality

### Benchmark Testing

```python
from comed import CoMedBenchmark

# Initialize benchmark system
benchmark = CoMedBenchmark(
    model_name="gpt-4o",
    api_key="your-key",
    api_base="https://api.openai.com/v1"
)

# Test drug combinations
drug_combinations = [
    ["warfarin", "aspirin"],
    ["metformin", "lisinopril"],
    ["atorvastatin", "amlodipine"]
]

# Run ablation study
ablation_results = benchmark.run_ablation_study(
    drug_combinations, retmax=20, verbose=True
)

# Save results
results_file = benchmark.save_benchmark_results(ablation_results)
print(f"Benchmark results saved to: {results_file}")
```

## 🏗️ Architecture Design

### Modular Design

```
CoMed v2.0
├── RAG Module (rag.py)
│   ├── Literature retrieval
│   ├── Relevance filtering
│   └── Statistics
├── CoT Module (cot.py)
│   ├── Chain-of-thought reasoning
│   ├── Step-by-step analysis
│   └── Result formatting
├── Multi-Agent Module (agents.py)
│   ├── Base agent class
│   ├── Specialized agents
│   └── Agent collaboration
├── Benchmark Module (benchmark.py)
│   ├── Ablation studies
│   ├── Performance evaluation
│   └── Report generation
└── Core Module (core.py)
    ├── Component integration
    ├── Configuration management
    └── Result aggregation
```

### Data Flow

```
Drug Combinations → RAG Retrieval → CoT Reasoning → Multi-Agent Analysis → Result Integration → Report Generation
    ↓                 ↓              ↓                ↓                    ↓
Literature Database  Association Analysis  Risk Assessment  Clinical Recommendations  Final Report
```

## 📚 API Reference

### Core Classes

- `CoMedData`: Main analysis class
- `RAGSystem`: RAG retrieval system
- `CoTReasoner`: CoT reasoning system
- `MultiAgentSystem`: Multi-agent system
- `CoMedBenchmark`: Benchmark testing system

### Key Methods

- `run_full_analysis()`: Run complete analysis pipeline
- `run_ablation_study()`: Run ablation study
- `run_component_test()`: Test individual component
- `compare_components()`: Compare component performance
- `set_config()`: Set configuration

### Environment Variables

- `MODEL_NAME`: Name of the LLM to use (e.g., "gpt-4o", "qwen2.5-32b-instruct")
- `API_BASE`: Base URL for the LLM API
- `API_KEY`: API key for LLM access
- `LOG_DIR`: Directory to store log files
- `OLD_OPENAI_API`: Whether to use the old OpenAI API format ("Yes" or "No")

## 🛠️ Development

### Adding New Components

```python
from comed.agents import Agent

class CustomAgent(Agent):
    def __init__(self, model_name, api_key, api_base):
        super().__init__("CustomAgent", "Custom Analysis", model_name, api_key, api_base)
    
    def _execute_task(self, input_data):
        # Implement custom analysis logic
        return {"custom_result": "Analysis result"}
```

### Custom Ablation Studies

```python
# Create custom benchmark test
class CustomBenchmark(CoMedBenchmark):
    def run_custom_ablation(self, drug_combinations):
        # Implement custom ablation study logic
        pass
```

## 🤝 Contributing

We welcome contributions in various forms:

1. **Code Contributions**: New features, bug fixes, performance optimizations
2. **Documentation Improvements**: Better examples, tutorials, API documentation
3. **Test Cases**: Unit tests, integration tests, benchmark tests
4. **Ablation Studies**: New evaluation metrics, test scenarios

## 📄 License

This project is licensed under the BSD License. See LICENSE file for details.

## 🙏 Acknowledgments

Thanks to the reviewers for their valuable feedback, which helped us build a more modular, evaluable framework.

## 📞 Contact

- Project Homepage: https://github.com/studentiz/comed
- Issue Reports: Please use GitHub Issues
- Email: studentiz@live.com

---

**Note**: This framework is for research purposes only and should not be used for clinical decision-making. Any medical decisions should be made in consultation with qualified healthcare professionals.