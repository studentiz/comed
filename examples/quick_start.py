#!/usr/bin/env python3
"""
Quick Start Example for CoMed
A simple example to get started with CoMed
"""

import os
import sys

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed

def main():
    """Quick start example"""
    print("🚀 CoMed Quick Start Example")
    print("="*50)
    
    # Step 1: Set up environment variables
    print("1️⃣ Setting up environment...")
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with your actual API key
    os.environ["LOG_DIR"] = "logs"
    
    print("✅ Environment configured")
    
    # Step 2: Initialize CoMed with drug list
    print("\n2️⃣ Initializing CoMed...")
    drugs = ["warfarin", "aspirin"]
    com = comed.CoMedData(drugs)
    
    print(f"✅ CoMed initialized with drugs: {', '.join(drugs)}")
    print(f"   Generated {len(com.drug_combinations)} drug combinations")
    
    # Step 3: Run analysis
    print("\n3️⃣ Running analysis...")
    print("   This may take a few minutes...")
    
    try:
        # Run full analysis pipeline
        report_path = com.run_full_analysis(retmax=30, verbose=True)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Report generated at: {report_path}")
        
        # Show some statistics
        if hasattr(com, 'papers') and not com.papers.empty:
            print(f"📚 Papers analyzed: {len(com.papers)}")
        
        if hasattr(com, 'associations') and not com.associations.empty:
            positive_count = len(com.associations[com.associations['Combined_medication'].str.lower() == 'yes'])
            print(f"🔍 Papers with drug combinations: {positive_count}")
        
        if hasattr(com, 'risk_analysis') and not com.risk_analysis.empty:
            print(f"⚠️ Risk analysis completed for {len(com.risk_analysis)} papers")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure you have set a valid API_KEY")
        print("   • Check your internet connection")
        print("   • Verify your API key has sufficient credits")
    
    print("\n🎯 Next steps:")
    print("   • Check the generated HTML report")
    print("   • Run the full demo: python examples/basic_demo.py")

if __name__ == "__main__":
    main()
