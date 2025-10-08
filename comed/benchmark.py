# benchmark.py
"""
Benchmark Testing and Evaluation Module
Supports ablation studies and performance evaluation
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

from .rag import RAGSystem
from .cot import CoTReasoner
from .agents import MultiAgentSystem

class CoMedBenchmark:
    """
    CoMed Benchmark Testing System
    Supports ablation studies and performance evaluation
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        """
        Initialize benchmark testing system
        
        Parameters
        ----------
        model_name : str
            LLM model name
        api_key : str, optional
            API key
        api_base : str, optional
            API base URL
        old_openai_api : str
            Whether to use old OpenAI API format
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.old_openai_api = old_openai_api
        self.logger = logging.getLogger('comed.benchmark')
        
        # Initialize components
        self.rag_system = RAGSystem()
        self.cot_reasoner = CoTReasoner(model_name, api_key, api_base, old_openai_api)
        self.multi_agent_system = MultiAgentSystem(model_name, api_key, api_base, old_openai_api)
        
        # Benchmark test results
        self.benchmark_results = {}
    
    def run_ablation_study(self, drug_combinations: List[List[str]], 
                          retmax: int = 30, verbose: bool = True) -> Dict[str, Any]:
        """
        Run ablation study
        
        Parameters
        ----------
        drug_combinations : List[List[str]]
            List of drug combinations
        retmax : int
            Maximum number of records per combination
        verbose : bool
            Whether to display progress
            
        Returns
        -------
        Dict[str, Any]
            Ablation study results
        """
        print(f"\n{'='*80}")
        print(f"🔬 Starting Ablation Study")
        print(f"{'='*80}")
        
        results = {}
        
        # 1. Baseline: RAG only
        print("\n📊 Baseline Test: RAG Retrieval Only")
        baseline_start = time.time()
        baseline_results = self.rag_system.search_drug_combinations(
            drug_combinations, retmax=retmax, verbose=verbose
        )
        baseline_time = time.time() - baseline_start
        baseline_stats = self.rag_system.get_retrieval_stats(baseline_results)
        
        results["baseline_rag"] = {
            "time": baseline_time,
            "stats": baseline_stats,
            "results": baseline_results
        }
        
        print(f"✓ Baseline test completed in {baseline_time:.1f} seconds")
        print(f"  • Papers retrieved: {baseline_stats['total_papers']}")
        
        # 2. RAG + CoT
        print("\n🧠 Test 1: RAG + CoT Reasoning")
        cot_start = time.time()
        
        if baseline_results.empty:
            print("⚠️ No baseline results, skipping CoT test")
            cot_results = pd.DataFrame()
        else:
            cot_results = self.cot_reasoner.batch_analyze_associations(
                baseline_results, verbose=verbose
            )
        
        cot_time = time.time() - cot_start
        cot_stats = self.cot_reasoner.get_reasoning_stats(cot_results)
        
        results["rag_cot"] = {
            "time": cot_time,
            "stats": cot_stats,
            "results": cot_results
        }
        
        print(f"✓ RAG + CoT test completed in {cot_time:.1f} seconds")
        print(f"  • Papers analyzed: {cot_stats['total_papers']}")
        print(f"  • Positive associations: {cot_stats['positive_associations']}")
        print(f"  • Positive rate: {cot_stats['positive_rate']:.2%}")
        
        # 3. RAG + CoT + Multi-Agent
        print("\n🤖 Test 2: RAG + CoT + Multi-Agent")
        agent_start = time.time()
        
        if cot_results.empty:
            print("⚠️ No CoT results, skipping multi-agent test")
            agent_results = pd.DataFrame()
        else:
            # Filter for positive associations
            positive_papers = cot_results[
                cot_results['Combined_medication'].str.lower() == 'yes'
            ]
            
            if positive_papers.empty:
                print("⚠️ No positive association papers, skipping multi-agent test")
                agent_results = pd.DataFrame()
            else:
                agent_results = self.multi_agent_system.batch_process(
                    positive_papers, verbose=verbose
                )
        
        agent_time = time.time() - agent_start
        agent_stats = self.multi_agent_system.get_agent_stats()
        
        results["full_system"] = {
            "time": agent_time,
            "stats": agent_stats,
            "results": agent_results
        }
        
        print(f"✓ Full system test completed in {agent_time:.1f} seconds")
        print(f"  • Multi-agent analyzed papers: {len(agent_results)}")
        
        # 4. Performance analysis
        performance_analysis = self._analyze_performance(results)
        results["performance_analysis"] = performance_analysis
        
        # 5. Generate ablation study report
        ablation_report = self._generate_ablation_report(results)
        results["ablation_report"] = ablation_report
        
        print(f"\n{'='*80}")
        print(f"✅ Ablation Study Completed")
        print(f"{'='*80}")
        
        return results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance metrics
        
        Parameters
        ----------
        results : Dict[str, Any]
            Test results
            
        Returns
        -------
        Dict[str, Any]
            Performance analysis results
        """
        analysis = {
            "time_comparison": {},
            "quality_metrics": {},
            "component_contributions": {},
            "efficiency_analysis": {}
        }
        
        # Time comparison
        if "baseline_rag" in results:
            analysis["time_comparison"]["rag_only"] = results["baseline_rag"]["time"]
        
        if "rag_cot" in results:
            analysis["time_comparison"]["rag_cot"] = results["rag_cot"]["time"]
        
        if "full_system" in results:
            analysis["time_comparison"]["full_system"] = results["full_system"]["time"]
        
        # Quality metrics
        if "rag_cot" in results and "baseline_rag" in results:
            rag_papers = results["baseline_rag"]["stats"]["total_papers"]
            cot_positive = results["rag_cot"]["stats"]["positive_associations"]
            
            if rag_papers > 0:
                analysis["quality_metrics"]["precision"] = cot_positive / rag_papers
                analysis["quality_metrics"]["recall"] = cot_positive / rag_papers
        
        # Component contribution analysis
        if "rag_cot" in results and "baseline_rag" in results:
            cot_contribution = (
                results["rag_cot"]["stats"]["positive_rate"] - 
                (results["baseline_rag"]["stats"]["total_papers"] / 
                 results["baseline_rag"]["stats"]["total_papers"] if 
                 results["baseline_rag"]["stats"]["total_papers"] > 0 else 0)
            )
            analysis["component_contributions"]["cot_contribution"] = cot_contribution
        
        if "full_system" in results and "rag_cot" in results:
            agent_contribution = (
                len(results["full_system"]["results"]) - 
                len(results["rag_cot"]["results"])
            )
            analysis["component_contributions"]["agent_contribution"] = agent_contribution
        
        # Efficiency analysis
        total_time = sum(analysis["time_comparison"].values())
        if total_time > 0:
            analysis["efficiency_analysis"]["rag_efficiency"] = (
                results["baseline_rag"]["stats"]["total_papers"] / 
                results["baseline_rag"]["time"] if results["baseline_rag"]["time"] > 0 else 0
            )
            
            if "rag_cot" in results:
                analysis["efficiency_analysis"]["cot_efficiency"] = (
                    results["rag_cot"]["stats"]["positive_associations"] / 
                    results["rag_cot"]["time"] if results["rag_cot"]["time"] > 0 else 0
                )
        
        return analysis
    
    def _generate_ablation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ablation study report
        
        Parameters
        ----------
        results : Dict[str, Any]
            Test results
            
        Returns
        -------
        Dict[str, Any]
            Ablation study report
        """
        report = {
            "study_summary": {
                "total_combinations": len(results.get("baseline_rag", {}).get("results", [])),
                "total_papers": results.get("baseline_rag", {}).get("stats", {}).get("total_papers", 0),
                "positive_associations": results.get("rag_cot", {}).get("stats", {}).get("positive_associations", 0)
            },
            "performance_summary": {
                "rag_time": results.get("baseline_rag", {}).get("time", 0),
                "cot_time": results.get("rag_cot", {}).get("time", 0),
                "agent_time": results.get("full_system", {}).get("time", 0)
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if "performance_analysis" in results:
            perf = results["performance_analysis"]
            
            if "component_contributions" in perf:
                cot_contrib = perf["component_contributions"].get("cot_contribution", 0)
                agent_contrib = perf["component_contributions"].get("agent_contribution", 0)
                
                if cot_contrib > 0.1:
                    report["recommendations"].append("CoT reasoning significantly improves analysis quality, recommend keeping")
                
                if agent_contrib > 0:
                    report["recommendations"].append("Multi-agent system provides additional value, recommend keeping")
                
                if cot_contrib < 0.05 and agent_contrib < 0:
                    report["recommendations"].append("Consider simplifying system architecture for efficiency")
        
        return report
    
    def run_component_ablation(self, drug_combinations: List[List[str]], 
                              retmax: int = 30, verbose: bool = True) -> Dict[str, Any]:
        """
        Run component ablation study
        
        Parameters
        ----------
        drug_combinations : List[List[str]]
            List of drug combinations
        retmax : int
            Maximum number of records per combination
        verbose : bool
            Whether to display progress
            
        Returns
        -------
        Dict[str, Any]
            Component ablation study results
        """
        print(f"\n{'='*80}")
        print(f"🔬 Starting Component Ablation Study")
        print(f"{'='*80}")
        
        results = {}
        
        # Test configurations
        test_configs = [
            {"name": "RAG Only", "rag": True, "cot": False, "agents": False},
            {"name": "RAG+CoT", "rag": True, "cot": True, "agents": False},
            {"name": "Full System", "rag": True, "cot": True, "agents": True}
        ]
        
        for config in test_configs:
            print(f"\n🧪 Test Configuration: {config['name']}")
            
            start_time = time.time()
            
            # Execute RAG
            if config["rag"]:
                rag_results = self.rag_system.search_drug_combinations(
                    drug_combinations, retmax=retmax, verbose=verbose
                )
            else:
                rag_results = pd.DataFrame()
            
            # Execute CoT
            if config["cot"] and not rag_results.empty:
                cot_results = self.cot_reasoner.batch_analyze_associations(
                    rag_results, verbose=verbose
                )
            else:
                cot_results = pd.DataFrame()
            
            # Execute multi-agent
            if config["agents"] and not cot_results.empty:
                positive_papers = cot_results[
                    cot_results['Combined_medication'].str.lower() == 'yes'
                ]
                if not positive_papers.empty:
                    agent_results = self.multi_agent_system.batch_process(
                        positive_papers, verbose=verbose
                    )
                else:
                    agent_results = pd.DataFrame()
            else:
                agent_results = pd.DataFrame()
            
            end_time = time.time()
            
            # Record results
            results[config["name"]] = {
                "time": end_time - start_time,
                "rag_results": rag_results,
                "cot_results": cot_results,
                "agent_results": agent_results,
                "config": config
            }
            
            print(f"✓ {config['name']} completed in {end_time - start_time:.1f} seconds")
        
        # Generate component ablation report
        component_report = self._generate_component_report(results)
        results["component_report"] = component_report
        
        return results
    
    def _generate_component_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate component ablation report
        
        Parameters
        ----------
        results : Dict[str, Any]
            Component test results
            
        Returns
        -------
        Dict[str, Any]
            Component ablation report
        """
        report = {
            "component_analysis": {},
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Analyze each component's contribution
        for config_name, result in results.items():
            if config_name == "component_report":
                continue
                
            analysis = {
                "time": result["time"],
                "papers_retrieved": len(result["rag_results"]),
                "papers_analyzed": len(result["cot_results"]),
                "papers_processed_by_agents": len(result["agent_results"])
            }
            
            report["component_analysis"][config_name] = analysis
        
        # Performance comparison
        if "RAG Only" in results and "RAG+CoT" in results:
            rag_time = results["RAG Only"]["time"]
            cot_time = results["RAG+CoT"]["time"]
            
            report["performance_comparison"]["cot_overhead"] = cot_time - rag_time
            report["performance_comparison"]["cot_efficiency"] = (
                len(results["RAG+CoT"]["cot_results"]) / cot_time if cot_time > 0 else 0
            )
        
        if "RAG+CoT" in results and "Full System" in results:
            cot_time = results["RAG+CoT"]["time"]
            full_time = results["Full System"]["time"]
            
            report["performance_comparison"]["agent_overhead"] = full_time - cot_time
            report["performance_comparison"]["agent_efficiency"] = (
                len(results["Full System"]["agent_results"]) / (full_time - cot_time) 
                if (full_time - cot_time) > 0 else 0
            )
        
        # Generate recommendations
        if "performance_comparison" in report:
            cot_overhead = report["performance_comparison"].get("cot_overhead", 0)
            agent_overhead = report["performance_comparison"].get("agent_overhead", 0)
            
            if cot_overhead < 10:  # Assume 10 seconds is acceptable
                report["recommendations"].append("CoT reasoning overhead is reasonable, recommend keeping")
            else:
                report["recommendations"].append("CoT reasoning overhead is large, consider optimization")
            
            if agent_overhead < 20:  # Assume 20 seconds is acceptable
                report["recommendations"].append("Multi-agent system overhead is reasonable, recommend keeping")
            else:
                report["recommendations"].append("Multi-agent system overhead is large, consider simplification")
        
        return report
    
    def save_benchmark_results(self, results: Dict[str, Any], 
                              filename: Optional[str] = None) -> str:
        """
        Save benchmark test results
        
        Parameters
        ----------
        results : Dict[str, Any]
            Test results
        filename : str, optional
            Filename
            
        Returns
        -------
        str
            Saved file path
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comed_benchmark_{timestamp}.json"
        
        # Convert DataFrame to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Benchmark test results saved to: {filename}")
        return filename
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to serializable format
        
        Parameters
        ----------
        obj : Any
            Object to convert
            
        Returns
        -------
        Any
            Serializable object
        """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj