#!/usr/bin/env python3
"""
Basic CoMed Demo
Demonstrates basic usage of CoMed for drug interaction analysis
"""

import os
import sys
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed

def setup_environment():
    """Setup environment variables for demo"""
    # Set LLM API configuration (replace with your actual API key)
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    os.environ["LOG_DIR"] = "logs"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

def demo_basic_usage():
    """Demo basic CoMed usage"""
    print("="*80)
    print("🔬 Basic CoMed Usage Demo")
    print("="*80)
    
    # Initialize CoMed with drug list
    drugs = ["warfarin", "aspirin"]
    print(f"Analyzing drug interactions for: {', '.join(drugs)}")
    
    com = comed.CoMedData(drugs)
    print(f"Generated {len(com.drug_combinations)} drug combinations")
    
    # Run full analysis pipeline
    print("\n🚀 Running full analysis pipeline...")
    try:
        report_path = com.run_full_analysis(retmax=5, verbose=True)
        print(f"\n✅ Analysis completed!")
        print(f"📄 Report generated at: {report_path}")
        return report_path
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Note: Make sure to set a valid API_KEY in the environment variables")
        return None

def demo_step_by_step():
    """Demo step-by-step analysis"""
    print("\n" + "="*80)
    print("📊 Step-by-Step Analysis Demo")
    print("="*80)
    
    # Initialize CoMed
    drugs = ["metformin", "lisinopril"]
    com = comed.CoMedData(drugs)
    
    print(f"Analyzing: {', '.join(drugs)}")
    
    try:
        # Step 1: Search literature
        print("\n📚 Step 1: Searching PubMed...")
        com.search(retmax=5, verbose=True)
        print(f"Found {len(com.papers)} papers")
        
        # Step 2: Analyze associations
        print("\n🧠 Step 2: Analyzing drug associations...")
        com.analyze_associations(verbose=True)
        positive_count = len(com.associations[com.associations['Combined_medication'].str.lower() == 'yes'])
        print(f"Found {positive_count} papers with drug combinations")
        
        # Step 3: Analyze risks
        print("\n⚠️ Step 3: Analyzing risks...")
        com.analyze_risks(verbose=True)
        print(f"Risk analysis completed for {len(com.risk_analysis)} papers")
        
        # Step 4: Generate report
        print("\n📄 Step 4: Generating report...")
        report_path = com.generate_report("Step_by_Step_Report.html", verbose=True)
        print(f"Report saved to: {report_path}")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Step-by-step analysis failed: {e}")
        return None

def demo_method_chaining():
    """Demo method chaining"""
    print("\n" + "="*80)
    print("🔗 Method Chaining Demo")
    print("="*80)
    
    # Initialize CoMed
    drugs = ["ibuprofen", "naproxen"]
    com = comed.CoMedData(drugs)
    
    print(f"Analyzing: {', '.join(drugs)}")
    
    try:
        # Use method chaining
        report_path = (com
                      .search(retmax=5, verbose=True)
                      .analyze_associations(verbose=True)
                      .analyze_risks(verbose=True)
                      .generate_report("Method_Chaining_Report.html", verbose=True))
        
        print(f"\n✅ Method chaining completed!")
        print(f"📄 Report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"❌ Method chaining failed: {e}")
        return None

def demo_incremental_drugs():
    """Demo adding drugs incrementally"""
    print("\n" + "="*80)
    print("➕ Incremental Drug Addition Demo")
    print("="*80)
    
    # Start with initial drugs
    initial_drugs = ["warfarin", "aspirin"]
    com = comed.CoMedData(initial_drugs)
    
    print(f"Initial drugs: {', '.join(initial_drugs)}")
    print(f"Initial combinations: {len(com.drug_combinations)}")
    
    # Add more drugs
    new_drugs = ["heparin", "clopidogrel"]
    com.add_drugs(new_drugs)
    
    print(f"Added drugs: {', '.join(new_drugs)}")
    print(f"Total drugs: {', '.join(com.drugs)}")
    print(f"Total combinations: {len(com.drug_combinations)}")
    
    try:
        # Run analysis
        report_path = com.run_full_analysis(retmax=3, verbose=True)
        print(f"\n✅ Incremental analysis completed!")
        print(f"📄 Report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"❌ Incremental analysis failed: {e}")
        return None

def demo_data_persistence():
    """Demo working with saved data"""
    print("\n" + "="*80)
    print("💾 Data Persistence Demo")
    print("="*80)
    
    # First, run an analysis to generate data files
    drugs = ["atorvastatin", "amlodipine"]
    com = comed.CoMedData(drugs)
    
    print(f"Running initial analysis for: {', '.join(drugs)}")
    
    try:
        # Run analysis to generate data files
        com.search(retmax=3, verbose=True)
        com.analyze_associations(verbose=True)
        
        # Save data files
        papers_file = "demo_papers.csv"
        associations_file = "demo_associations.csv"
        
        com.papers.to_csv(papers_file, index=False)
        com.associations.to_csv(associations_file, index=False)
        
        print(f"📁 Data saved to: {papers_file}, {associations_file}")
        
        # Now load the data back
        print("\n📂 Loading saved data...")
        import comed.io as io
        
        loaded_papers = io.read_papers(papers_file)
        loaded_associations = io.read_associations(associations_file)
        
        print(f"Loaded {len(loaded_papers)} papers and {len(loaded_associations)} associations")
        
        # Create new CoMed instance with loaded data
        new_com = comed.CoMedData()
        new_com.papers = loaded_papers
        new_com.associations = loaded_associations
        
        # Continue analysis from where we left off
        print("\n🔄 Continuing analysis with loaded data...")
        report_path = new_com.analyze_risks(verbose=True).generate_report("Persistence_Demo_Report.html", verbose=True)
        
        print(f"\n✅ Data persistence demo completed!")
        print(f"📄 Report saved to: {report_path}")
        
        # Clean up demo files
        import os
        if os.path.exists(papers_file):
            os.remove(papers_file)
        if os.path.exists(associations_file):
            os.remove(associations_file)
        print("🧹 Cleaned up demo files")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Data persistence demo failed: {e}")
        return None


def main():
    """Main demo function"""
    print("🚀 CoMed Comprehensive Demo")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    print("\n⚠️  Note: This demo requires a valid API key to run successfully.")
    print("   Set your API_KEY environment variable before running.")
    print("   Some demos may fail without a valid API key.\n")
    
    # Run different demos
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Step-by-Step Analysis", demo_step_by_step),
        ("Method Chaining", demo_method_chaining),
        ("Incremental Drug Addition", demo_incremental_drugs),
        ("Data Persistence", demo_data_persistence)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            result = demo_func()
            results[demo_name] = "✅ Success" if result else "❌ Failed"
        except Exception as e:
            print(f"❌ {demo_name} failed with error: {e}")
            results[demo_name] = f"❌ Error: {str(e)[:50]}..."
    
    # Summary
    print("\n" + "="*80)
    print("📋 Demo Results Summary")
    print("="*80)
    
    for demo_name, status in results.items():
        print(f"  • {demo_name}: {status}")
    
    print(f"\n🎯 Total demos: {len(demos)}")
    print(f"✅ Successful: {sum(1 for status in results.values() if '✅' in status)}")
    print(f"❌ Failed: {sum(1 for status in results.values() if '❌' in status)}")
    
    print("\n💡 Tips:")
    print("  • Make sure to set a valid API_KEY environment variable")
    print("  • Check your internet connection for PubMed access")
    print("  • Some demos may take a few minutes to complete")

if __name__ == "__main__":
    main()
