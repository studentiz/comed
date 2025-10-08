#!/usr/bin/env python3
"""
Load and Analyze Existing Data
Load existing association data and run multi-agent analysis
"""

import os
import sys
import pandas as pd
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed

def setup_environment():
    """Setup environment variables"""
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    os.environ["LOG_DIR"] = "logs"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

def load_and_analyze_existing_data():
    """Load existing data and run analysis"""
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
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def run_multi_agent_analysis(associations_df):
    """Run multi-agent analysis on existing data"""
    print("\n" + "="*80)
    print("🤖 Multi-Agent Analysis")
    print("="*80)
    
    # Filter for positive associations
    positive_papers = associations_df[
        associations_df['Combined_medication'].str.lower() == 'yes'
    ].copy()
    
    if positive_papers.empty:
        print("⚠️ No positive associations found!")
        print("Cannot run multi-agent analysis without positive associations.")
        return None
    
    print(f"📋 Running analysis on {len(positive_papers)} papers with positive associations")
    
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
        print("\n🔄 Processing papers with multi-agent system...")
        print("This may take a few minutes...")
        
        agent_results = agent_system.batch_process(
            positive_papers, verbose=True
        )
        
        print(f"\n✅ Multi-agent analysis completed!")
        print(f"📊 Results: {len(agent_results)} papers processed")
        
        # Show results
        if not agent_results.empty:
            print(f"\n📋 Results preview:")
            print(f"  • Columns: {list(agent_results.columns)}")
            print(f"  • Sample result keys: {list(agent_results.iloc[0].keys()) if len(agent_results) > 0 else 'No results'}")
        
        # Save results
        output_file = "multi_agent_analysis_results.csv"
        agent_results.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Show agent statistics
        stats = agent_system.get_agent_stats()
        print(f"\n🤖 Agent Statistics:")
        for agent_name, agent_info in stats.items():
            print(f"  • {agent_name}: {agent_info['status']}, Conversations: {agent_info['conversation_count']}")
        
        return agent_results
        
    except Exception as e:
        print(f"❌ Multi-agent analysis failed: {e}")
        return None

def run_risk_analysis(associations_df):
    """Run risk analysis on existing data"""
    print("\n" + "="*80)
    print("⚠️ Risk Analysis")
    print("="*80)
    
    # Filter for positive associations
    positive_papers = associations_df[
        associations_df['Combined_medication'].str.lower() == 'yes'
    ].copy()
    
    if positive_papers.empty:
        print("⚠️ No positive associations found!")
        return None
    
    print(f"📋 Running risk analysis on {len(positive_papers)} papers")
    
    try:
        # Initialize CoMed system
        print("\n🔄 Initializing CoMed system...")
        com = comed.CoMedData()
        com.associations = positive_papers
        
        # Run risk analysis
        print("\n🔄 Running risk analysis...")
        com.analyze_risks(verbose=True)
        
        print(f"\n✅ Risk analysis completed!")
        print(f"📊 Results: {len(com.risk_analysis)} papers analyzed")
        
        # Save results
        output_file = "risk_analysis_results.csv"
        com.risk_analysis.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        return com.risk_analysis
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {e}")
        return None

def generate_report(risk_analysis_df):
    """Generate HTML report"""
    print("\n" + "="*80)
    print("📄 Generating Report")
    print("="*80)
    
    if risk_analysis_df is None or risk_analysis_df.empty:
        print("⚠️ No risk analysis data available!")
        return None
    
    try:
        # Initialize CoMed system
        print("\n🔄 Initializing CoMed system...")
        com = comed.CoMedData()
        com.risk_analysis = risk_analysis_df
        
        # Generate report
        print("\n🔄 Generating HTML report...")
        report_path = com.generate_report("Loaded_Data_Report.html", verbose=True)
        
        print(f"\n✅ Report generated successfully!")
        print(f"📄 Report saved to: {report_path}")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None

def main():
    """Main function"""
    print("🚀 Load and Analyze Existing Data")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    print("\n⚠️  Note: This test requires a valid API key to run successfully.")
    print("   Set your API_KEY environment variable before running.")
    print("   The test will use existing association data from ddc_papers_association_pd.csv\n")
    
    # Load existing data
    associations_df = load_and_analyze_existing_data()
    if associations_df is None:
        print("❌ Cannot proceed without association data")
        return
    
    # Run multi-agent analysis
    print("\n1️⃣ Multi-Agent Analysis")
    agent_results = run_multi_agent_analysis(associations_df)
    
    # Run risk analysis
    print("\n2️⃣ Risk Analysis")
    risk_results = run_risk_analysis(associations_df)
    
    # Generate report
    print("\n3️⃣ Report Generation")
    report_path = generate_report(risk_results)
    
    # Summary
    print("\n" + "="*80)
    print("📋 Analysis Summary")
    print("="*80)
    
    if agent_results is not None:
        print("✅ Multi-agent analysis: COMPLETED")
        print(f"   • Processed {len(agent_results)} papers")
        print(f"   • Results saved to: multi_agent_analysis_results.csv")
    else:
        print("❌ Multi-agent analysis: FAILED")
    
    if risk_results is not None:
        print("✅ Risk analysis: COMPLETED")
        print(f"   • Analyzed {len(risk_results)} papers")
        print(f"   • Results saved to: risk_analysis_results.csv")
    else:
        print("❌ Risk analysis: FAILED")
    
    if report_path:
        print("✅ Report generation: COMPLETED")
        print(f"   • Report saved to: {report_path}")
    else:
        print("❌ Report generation: FAILED")
    
    print("\n🎯 Next steps:")
    print("  • Check the generated results files")
    print("  • Review the HTML report")
    print("  • Analyze the multi-agent collaboration patterns")

if __name__ == "__main__":
    main()
