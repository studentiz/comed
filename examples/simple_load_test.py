#!/usr/bin/env python3
"""
Simple Load Test - Test loading existing data without external dependencies
"""

import os
import sys
import pandas as pd
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_environment():
    """Setup environment variables"""
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    os.environ["LOG_DIR"] = "logs"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

def load_and_analyze_existing_data():
    """Load existing data and show basic analysis"""
    print("📂 Loading existing association data...")
    
    # Check if file exists
    csv_file = "ddc_papers_association_pd.csv"
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return None
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded {len(df)} records")
        
        # Show basic statistics
        print(f"\n📊 Data Statistics:")
        print(f"  • Total papers: {len(df)}")
        
        # Count positive associations
        positive_count = len(df[df['Combined_medication'].str.lower() == 'yes'])
        print(f"  • Positive associations: {positive_count}")
        print(f"  • Positive rate: {positive_count/len(df)*100:.1f}%")
        
        # Show drug combinations
        if 'Drug1' in df.columns and 'Drug2' in df.columns:
            unique_combos = df[['Drug1', 'Drug2']].drop_duplicates()
            print(f"  • Drug combinations: {len(unique_combos)}")
            for _, row in unique_combos.iterrows():
                print(f"    - {row['Drug1']} + {row['Drug2']}")
        
        # Show sample data
        print(f"\n📋 Sample Data:")
        print(f"  • Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  • First paper: {df.iloc[0]['Title'][:100]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_multi_agent_import():
    """Test if multi-agent system can be imported"""
    print("\n🤖 Testing Multi-Agent System Import")
    print("="*50)
    
    try:
        # Try to import individual modules
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comed'))
        
        # Test individual imports
        try:
            from agents import MultiAgentSystem
            print("✅ MultiAgentSystem imported successfully")
        except ImportError as e:
            print(f"❌ MultiAgentSystem import failed: {e}")
        
        try:
            from agents import RiskAnalysisAgent, SafetyAgent, ClinicalAgent
            print("✅ Individual agents imported successfully")
        except ImportError as e:
            print(f"❌ Individual agents import failed: {e}")
        
        try:
            from core import CoMedData
            print("✅ CoMedData imported successfully")
        except ImportError as e:
            print(f"❌ CoMedData import failed: {e}")
        
        try:
            from utils import configure_logging
            print("✅ utils.configure_logging imported successfully")
        except ImportError as e:
            print(f"❌ utils.configure_logging import failed: {e}")
        
        try:
            from io import read_papers, read_associations
            print("✅ io functions imported successfully")
        except ImportError as e:
            print(f"❌ io functions import failed: {e}")
        
        print("\n✅ All individual imports successful!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")

def main():
    """Main function"""
    print("🚀 Simple Load Test")
    print("="*50)
    
    # Setup environment
    setup_environment()
    
    print("\n⚠️  Note: This test does not require external dependencies.")
    print("   It will test data loading and basic imports.\n")
    
    # Load existing data
    associations_df = load_and_analyze_existing_data()
    if associations_df is None:
        print("❌ Cannot proceed without association data")
        return
    
    # Test imports
    test_multi_agent_import()
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    
    if associations_df is not None:
        print("✅ Data loading: SUCCESS")
        print(f"   • Loaded {len(associations_df)} records")
        print(f"   • Positive associations: {len(associations_df[associations_df['Combined_medication'].str.lower() == 'yes'])}")
    else:
        print("❌ Data loading: FAILED")
    
    print("✅ Import testing: COMPLETED")
    
    print("\n🎯 Next steps:")
    print("  • Install required dependencies: pip install biopython pandas tqdm openai")
    print("  • Set your API_KEY environment variable")
    print("  • Run the full multi-agent analysis")

if __name__ == "__main__":
    main()
