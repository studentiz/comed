#!/usr/bin/env python3
"""
Direct Multi-Agent Test
Test multi-agent system directly using existing association data
"""

import os
import sys
import pandas as pd
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comed
from comed import MultiAgentSystem, CoMedData

def setup_environment():
    """Setup environment variables"""
    # Set LLM API configuration
    os.environ["MODEL_NAME"] = "gpt-4o"
    os.environ["API_BASE"] = "https://api.openai.com/v1"
    os.environ["API_KEY"] = "your-api-key-here"  # Replace with actual API key
    os.environ["LOG_DIR"] = "logs"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

def load_existing_data():
    """Load existing association data"""
    print("📂 Loading existing association data...")
    
    # Check if the file exists
    csv_file = "ddc_papers_association_pd.csv"
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return None
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded {len(df)} records from {csv_file}")
        
        # Show basic statistics
        print(f"📊 Data Statistics:")
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

def test_multi_agent_system(associations_df):
    """Test multi-agent system with existing data"""
    print("\n" + "="*80)
    print("🤖 Testing Multi-Agent System")
    print("="*80)
    
    # Filter for positive associations only
    positive_papers = associations_df[
        associations_df['Combined_medication'].str.lower() == 'yes'
    ].copy()
    
    if positive_papers.empty:
        print("⚠️ No positive associations found!")
        print("Cannot test multi-agent system without positive associations.")
        return None
    
    print(f"📋 Testing with {len(positive_papers)} papers with positive associations")
    
    # Initialize multi-agent system
    try:
        agent_system = MultiAgentSystem(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        print("✅ Multi-agent system initialized successfully")
        
        # Process papers with multi-agent system
        print("\n🔄 Processing papers with multi-agent system...")
        print("This may take a few minutes...")
        
        agent_results = agent_system.batch_process(
            positive_papers, verbose=True
        )
        
        print(f"\n✅ Multi-agent processing completed!")
        print(f"📊 Results: {len(agent_results)} papers processed")
        
        # Show agent statistics
        agent_stats = agent_system.get_agent_stats()
        print(f"\n🤖 Agent Statistics:")
        for agent_name, stats in agent_stats.items():
            print(f"  • {agent_name}: {stats['status']}, Conversations: {stats['conversation_count']}")
        
        # Save results
        output_file = "multi_agent_results.csv"
        agent_results.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        return agent_results
        
    except Exception as e:
        print(f"❌ Multi-agent system test failed: {e}")
        return None

def test_individual_agents(associations_df):
    """Test individual agents"""
    print("\n" + "="*80)
    print("🧪 Testing Individual Agents")
    print("="*80)
    
    # Filter for positive associations
    positive_papers = associations_df[
        associations_df['Combined_medication'].str.lower() == 'yes'
    ].copy()
    
    if positive_papers.empty:
        print("⚠️ No positive associations found!")
        return None
    
    # Test with first few papers
    test_papers = positive_papers.head(3)
    print(f"📋 Testing with {len(test_papers)} papers")
    
    try:
        from comed import RiskAnalysisAgent, SafetyAgent, ClinicalAgent
        
        # Initialize individual agents
        risk_agent = RiskAnalysisAgent(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        safety_agent = SafetyAgent(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        clinical_agent = ClinicalAgent(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        print("✅ Individual agents initialized successfully")
        
        # Test each agent
        for idx, (_, paper) in enumerate(test_papers.iterrows()):
            print(f"\n📄 Testing Paper {idx+1}: {paper['Drug1']} + {paper['Drug2']}")
            
            # Prepare task data
            task_data = {
                "drug1": paper['Drug1'],
                "drug2": paper['Drug2'],
                "abstract": paper['Abstract'],
                "title": paper['Title']
            }
            
            # Test Risk Analysis Agent
            print("  🔍 Risk Analysis Agent...")
            try:
                risk_result = risk_agent.process_task(task_data)
                print(f"    ✅ Risk analysis completed")
                print(f"    📊 Result keys: {list(risk_result.keys())}")
            except Exception as e:
                print(f"    ❌ Risk analysis failed: {e}")
            
            # Test Safety Agent
            print("  🛡️ Safety Agent...")
            try:
                safety_result = safety_agent.process_task(task_data)
                print(f"    ✅ Safety analysis completed")
                print(f"    📊 Result keys: {list(safety_result.keys())}")
            except Exception as e:
                print(f"    ❌ Safety analysis failed: {e}")
            
            # Test Clinical Agent
            print("  🏥 Clinical Agent...")
            try:
                clinical_result = clinical_agent.process_task(task_data)
                print(f"    ✅ Clinical analysis completed")
                print(f"    📊 Result keys: {list(clinical_result.keys())}")
            except Exception as e:
                print(f"    ❌ Clinical analysis failed: {e}")
        
        print(f"\n✅ Individual agent testing completed!")
        
    except Exception as e:
        print(f"❌ Individual agent testing failed: {e}")

def test_agent_communication():
    """Test agent-to-agent communication"""
    print("\n" + "="*80)
    print("💬 Testing Agent Communication")
    print("="*80)
    
    try:
        from comed import RiskAnalysisAgent, SafetyAgent, ClinicalAgent
        
        # Initialize agents
        risk_agent = RiskAnalysisAgent(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        safety_agent = SafetyAgent(
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE")
        )
        
        print("✅ Agents initialized successfully")
        
        # Test communication
        print("\n💬 Testing agent-to-agent communication...")
        
        # Risk agent sends message to safety agent
        message = "I found significant bleeding risk with this drug combination. Please evaluate safety profile."
        risk_agent.send_message(safety_agent, message)
        
        # Safety agent responds
        response = "I'll evaluate the safety profile. Based on my analysis, this combination requires careful monitoring."
        safety_agent.send_message(risk_agent, response)
        
        # Check conversation history
        print(f"\n📝 Risk Agent conversation history: {len(risk_agent.conversation_history)} messages")
        print(f"📝 Safety Agent conversation history: {len(safety_agent.conversation_history)} messages")
        
        # Show recent messages
        if risk_agent.conversation_history:
            print(f"\n🔍 Recent Risk Agent messages:")
            for msg in risk_agent.conversation_history[-2:]:
                print(f"  • {msg['role']}: {msg['content'][:100]}...")
        
        if safety_agent.conversation_history:
            print(f"\n🔍 Recent Safety Agent messages:")
            for msg in safety_agent.conversation_history[-2:]:
                print(f"  • {msg['role']}: {msg['content'][:100]}...")
        
        print(f"\n✅ Agent communication test completed!")
        
    except Exception as e:
        print(f"❌ Agent communication test failed: {e}")

def main():
    """Main function"""
    print("🚀 Direct Multi-Agent Test")
    print("="*80)
    
    # Setup environment
    setup_environment()
    
    print("\n⚠️  Note: This test requires a valid API key to run successfully.")
    print("   Set your API_KEY environment variable before running.")
    print("   The test will use existing association data from ddc_papers_association_pd.csv\n")
    
    # Load existing data
    associations_df = load_existing_data()
    if associations_df is None:
        print("❌ Cannot proceed without association data")
        return
    
    # Test multi-agent system
    print("\n1️⃣ Testing Multi-Agent System")
    agent_results = test_multi_agent_system(associations_df)
    
    # Test individual agents
    print("\n2️⃣ Testing Individual Agents")
    test_individual_agents(associations_df)
    
    # Test agent communication
    print("\n3️⃣ Testing Agent Communication")
    test_agent_communication()
    
    # Summary
    print("\n" + "="*80)
    print("📋 Test Summary")
    print("="*80)
    
    if agent_results is not None:
        print("✅ Multi-agent system test: PASSED")
        print(f"   • Processed {len(agent_results)} papers")
        print(f"   • Results saved to: multi_agent_results.csv")
    else:
        print("❌ Multi-agent system test: FAILED")
    
    print("✅ Individual agent test: COMPLETED")
    print("✅ Agent communication test: COMPLETED")
    
    print("\n🎯 Next steps:")
    print("  • Check the generated results files")
    print("  • Review agent conversation logs")
    print("  • Analyze multi-agent collaboration patterns")

if __name__ == "__main__":
    main()
