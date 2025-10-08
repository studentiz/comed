#!/usr/bin/env python3
"""
Simple Multi-Agent Test
Quick test of multi-agent system using existing association data
"""

import os
import sys
import pandas as pd

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed

def main():
    """Simple multi-agent test"""
    print("🤖 Simple Multi-Agent Test")
    print("="*50)
    
    # Setup environment
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    
    # Load existing association data
    csv_file = "ddc_papers_association_pd.csv"
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    print(f"📂 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} records")
    
    # Filter for positive associations
    positive_papers = df[df['Combined_medication'].str.lower() == 'yes']
    print(f"📊 Found {len(positive_papers)} papers with positive associations")
    
    if positive_papers.empty:
        print("⚠️ No positive associations found!")
        return
    
    # Test with first 2 papers
    test_papers = positive_papers.head(2)
    print(f"🧪 Testing with {len(test_papers)} papers")
    
    try:
        # Initialize multi-agent system
        print("\n🔄 Initializing multi-agent system...")
        agent_system = comed.MultiAgentSystem(
            model_name=os.getenv("MODEL_NAME"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        print("✅ Multi-agent system initialized")
        
        # Process papers
        print("\n🔄 Processing papers...")
        results = agent_system.batch_process(test_papers, verbose=True)
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Results: {len(results)} papers processed")
        
        # Show results
        if not results.empty:
            print(f"\n📋 Results preview:")
            print(f"  • Columns: {list(results.columns)}")
            print(f"  • Sample result keys: {list(results.iloc[0].keys()) if len(results) > 0 else 'No results'}")
        
        # Save results
        output_file = "simple_agent_results.csv"
        results.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Show agent statistics
        stats = agent_system.get_agent_stats()
        print(f"\n🤖 Agent Statistics:")
        for agent_name, agent_info in stats.items():
            print(f"  • {agent_name}: {agent_info['status']}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  • Make sure you have set a valid API_KEY")
        print("  • Check your internet connection")
        print("  • Verify your API key has sufficient credits")

if __name__ == "__main__":
    main()
