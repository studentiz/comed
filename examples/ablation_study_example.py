#!/usr/bin/env python3
"""
Ablation Study Example
Demonstrates how to use CoMed for ablation studies and component testing
"""

import os
import sys
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed
from comed import CoMedData, CoMedBenchmark

def setup_environment():
    """Setup environment variables"""
    # Set LLM API configuration
    os.environ["MODEL_NAME"] = "gpt-4o"  # or use other models
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

def run_basic_ablation_study():
    """Run basic ablation study"""
    print("="*80)
    print("🔬 Basic Ablation Study Example")
    print("="*80)
    
    # Initialize CoMed system
    drugs = ["warfarin", "aspirin", "ibuprofen"]
    com = CoMedData(drugs)
    
    print(f"Analyzing drugs: {', '.join(drugs)}")
    print(f"Drug combinations: {len(com.drug_combinations)}")
    
    # Run ablation study
    ablation_results = com.run_ablation_study(retmax=10, verbose=True)
    
    print("\n📊 Ablation Study Results:")
    print(f"• Results file: {ablation_results.get('results_file', 'N/A')}")
    
    return ablation_results

def run_component_comparison():
    """Run component comparison"""
    print("\n" + "="*80)
    print("📊 Component Performance Comparison Example")
    print("="*80)
    
    # Initialize CoMed system
    drugs = ["metformin", "lisinopril"]
    com = CoMedData(drugs)
    
    print(f"Analyzing drugs: {', '.join(drugs)}")
    
    # Compare component performance
    comparison_results = com.compare_components(retmax=5, verbose=True)
    
    print("\n📈 Component Comparison Results:")
    if "comparison_report" in comparison_results:
        report = comparison_results["comparison_report"]
        
        print("Performance Summary:")
        for component, perf in report.get("performance_summary", {}).items():
            print(f"  • {component.upper()}: {perf['time']:.1f}s, Efficiency: {perf['efficiency']:.2f}")
        
        print("\nRecommendations:")
        for rec in report.get("recommendations", []):
            print(f"  • {rec}")
    
    return comparison_results

def run_individual_component_tests():
    """Run individual component tests"""
    print("\n" + "="*80)
    print("🧪 Individual Component Testing Example")
    print("="*80)
    
    # Initialize CoMed system
    drugs = ["atorvastatin", "amlodipine"]
    com = CoMedData(drugs)
    
    print(f"Analyzing drugs: {', '.join(drugs)}")
    
    # Test RAG component
    print("\n📚 Testing RAG Component...")
    rag_result = com.run_component_test("rag", retmax=5, verbose=True)
    print(f"RAG Result: {rag_result['time']:.1f}s, Retrieved {rag_result['stats']['total_papers']} papers")
    
    # Test CoT component
    print("\n🧠 Testing CoT Component...")
    cot_result = com.run_component_test("cot", retmax=5, verbose=True)
    print(f"CoT Result: {cot_result['time']:.1f}s, Analyzed {cot_result['stats']['total_papers']} papers")
    
    # Test multi-agent component
    print("\n🤖 Testing Multi-Agent Component...")
    agent_result = com.run_component_test("agents", retmax=5, verbose=True)
    print(f"Multi-Agent Result: {agent_result['time']:.1f}s, Processed {len(agent_result['results'])} papers")
    
    return {
        "rag": rag_result,
        "cot": cot_result,
        "agents": agent_result
    }

def run_benchmark_study():
    """Run benchmark study"""
    print("\n" + "="*80)
    print("🏆 Benchmark Study Example")
    print("="*80)
    
    # Initialize benchmark system
    benchmark = CoMedBenchmark(
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE")
    )
    
    # Test drug combinations
    drug_combinations = [
        ["warfarin", "aspirin"],
        ["metformin", "lisinopril"],
        ["atorvastatin", "amlodipine"]
    ]
    
    print(f"Testing {len(drug_combinations)} drug combinations")
    
    # Run ablation study
    ablation_results = benchmark.run_ablation_study(
        drug_combinations, retmax=5, verbose=True
    )
    
    # Run component ablation study
    component_results = benchmark.run_component_ablation(
        drug_combinations, retmax=5, verbose=True
    )
    
    # Save results
    results_file = benchmark.save_benchmark_results(ablation_results)
    print(f"\n📄 Benchmark results saved to: {results_file}")
    
    return {
        "ablation_results": ablation_results,
        "component_results": component_results,
        "results_file": results_file
    }

def run_modular_usage_example():
    """Run modular usage example"""
    print("\n" + "="*80)
    print("🔗 Modular Usage Example")
    print("="*80)
    
    from comed import RAGSystem, CoTReasoner, MultiAgentSystem
    
    # Use only RAG
    print("\n📚 Using RAG System Only...")
    rag_system = RAGSystem()
    papers = rag_system.search_drug_combinations([["warfarin", "aspirin"]], retmax=5, verbose=True)
    print(f"RAG retrieved {len(papers)} papers")
    
    # Use only CoT
    print("\n🧠 Using CoT System Only...")
    cot_reasoner = CoTReasoner(
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE")
    )
    cot_results = cot_reasoner.batch_analyze_associations(papers, verbose=True)
    print(f"CoT analyzed {len(cot_results)} papers")
    
    # Use only multi-agent system
    print("\n🤖 Using Multi-Agent System Only...")
    agent_system = MultiAgentSystem(
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE")
    )
    
    # Filter positive associations
    positive_papers = cot_results[cot_results['Combined_medication'].str.lower() == 'yes']
    if not positive_papers.empty:
        agent_results = agent_system.batch_process(positive_papers, verbose=True)
        print(f"Multi-agent processed {len(agent_results)} papers")
    else:
        print("No positive associations found for multi-agent processing")
    
    return {
        "rag_papers": papers,
        "cot_results": cot_results,
        "agent_results": agent_results if not positive_papers.empty else None
    }

def main():
    """Main function"""
    print("🚀 CoMed Ablation Study Examples")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    try:
        # 1. Basic ablation study
        print("\n1️⃣ Basic Ablation Study")
        ablation_results = run_basic_ablation_study()
        
        # 2. Component comparison
        print("\n2️⃣ Component Performance Comparison")
        comparison_results = run_component_comparison()
        
        # 3. Individual component tests
        print("\n3️⃣ Individual Component Tests")
        component_tests = run_individual_component_tests()
        
        # 4. Benchmark study
        print("\n4️⃣ Benchmark Study")
        benchmark_results = run_benchmark_study()
        
        # 5. Modular usage example
        print("\n5️⃣ Modular Usage Example")
        modular_results = run_modular_usage_example()
        
        print("\n" + "="*80)
        print("✅ All Examples Completed Successfully!")
        print("="*80)
        
        print("\n📋 Results Summary:")
        print(f"• Ablation Study: {'Completed' if ablation_results else 'Failed'}")
        print(f"• Component Comparison: {'Completed' if comparison_results else 'Failed'}")
        print(f"• Component Tests: {'Completed' if component_tests else 'Failed'}")
        print(f"• Benchmark Study: {'Completed' if benchmark_results else 'Failed'}")
        print(f"• Modular Usage: {'Completed' if modular_results else 'Failed'}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        logging.error(f"Example execution failed: {e}")

if __name__ == "__main__":
    main()