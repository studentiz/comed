# core.py
"""
CoMed Core Module
Main interface for drug co-medication risk analysis with modular architecture
"""

import pandas as pd
import logging
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import sys

# Import modular components
from .rag import RAGSystem
from .cot import CoTReasoner
from .agents import MultiAgentSystem
from .benchmark import CoMedBenchmark

class CoMedData:
    """
    Core data structure for CoMed package.
    Stores drug combinations, literature information, and analysis results.
    
    Parameters
    ----------
    drugs : List[str], optional
        List of drug names to analyze
    """
    
    def __init__(self, drugs: Optional[List[str]] = None):
        # Store original drug list
        self.drugs = drugs if drugs is not None else []
        
        # Configuration settings
        self.config = {
            'model_name': os.getenv('MODEL_NAME', ''),
            'api_base': os.getenv('API_BASE', ''),
            'api_key': os.getenv('API_KEY', ''),
            'log_dir': os.getenv('LOG_DIR', 'logs'),
            'old_openai_api': os.getenv('OLD_OPENAI_API', 'No')
        }
        
        # Initialize logger
        self._setup_logging()
        
        # Generate drug combinations if drugs provided
        self.drug_combinations = []
        if drugs is not None and len(drugs) > 1:
            self._generate_drug_combinations()
        
        # Initialize data containers
        self.papers = pd.DataFrame()
        self.associations = pd.DataFrame()
        self.risk_analysis = pd.DataFrame()
    
    def _generate_drug_combinations(self):
        """Generate all possible pairwise drug combinations."""
        drugs_count = len(self.drugs)
        for i in range(drugs_count):
            for j in range(i+1, drugs_count):
                self.drug_combinations.append([self.drugs[i], self.drugs[j]])
        
        self.logger.info(f"Generated {len(self.drug_combinations)} drug combinations")
        print(f"✓ Generated {len(self.drug_combinations)} drug combinations:")
        for i, combo in enumerate(self.drug_combinations):
            print(f"  {i+1}. {combo[0]} + {combo[1]}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('comed')
        self.logger.setLevel(logging.INFO)
        
        # Check if handlers already exist
        if not self.logger.handlers:
            # Create file handler
            log_filename = f"comed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
            
            # Also add a console handler for minimal info
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter('INFO:comed:%(message)s'))
            self.logger.addHandler(console_handler)
            
            self.logger.info("CoMed analysis started")
            self.logger.info(f"Drug list: {self.drugs}")
    
    def set_config(self, config: Dict[str, Any]) -> 'CoMedData':
        """
        Update configuration settings.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary of configuration settings to update
            
        Returns
        -------
        CoMedData
            Self for method chaining
        """
        self.config.update(config)
        return self
    
    def add_drugs(self, drugs: List[str]) -> 'CoMedData':
        """
        Add additional drugs to the analysis.
        
        Parameters
        ----------
        drugs : List[str]
            List of drug names to add
            
        Returns
        -------
        CoMedData
            Self for method chaining
        """
        # Add new drugs to list
        print(f"\n{'='*80}\n📊 ADDING NEW DRUGS: {', '.join(drugs)}\n{'='*80}")
        
        new_drugs = [drug for drug in drugs if drug not in self.drugs]
        if not new_drugs:
            print("No new drugs to add. All specified drugs are already in the list.")
            return self
            
        self.drugs.extend(new_drugs)
        
        original_combo_count = len(self.drug_combinations)
        
        # Generate new combinations
        new_combinations = []
        for new_drug in new_drugs:
            for existing_drug in self.drugs[:-len(new_drugs)]:
                new_combinations.append([existing_drug, new_drug])
        
        # Also add combinations between new drugs
        drugs_count = len(new_drugs)
        for i in range(drugs_count):
            for j in range(i+1, drugs_count):
                new_combinations.append([new_drugs[i], new_drugs[j]])
        
        self.drug_combinations.extend(new_combinations)
        
        # Print out the new combinations
        print(f"✓ Added {len(new_combinations)} new drug combinations:")
        for i, combo in enumerate(new_combinations):
            print(f"  {original_combo_count+i+1}. {combo[0]} + {combo[1]}")
        
        self.logger.info(f"Added {len(new_drugs)} drugs. Total combinations: {len(self.drug_combinations)}")
        return self
    
    def search(self, retmax: int = 30, email: str = 'your_email@example.com', 
               retry: int = 3, delay: int = 3, filepath: str = "ddc_papers.csv", 
               verbose: bool = True) -> 'CoMedData':
        """
        Search papers for all drug combinations using RAG system.
        
        Parameters
        ----------
        retmax : int, default=30
            Maximum number of records to retrieve per combination
        email : str, default='your_email@example.com'
            Email address for Entrez API
        retry : int, default=3
            Maximum number of retry attempts
        delay : int, default=3
            Delay between requests in seconds
        filepath : str, default="ddc_papers.csv"
            Path to save the search results
        verbose : bool, default=True
            Whether to display progress bar
            
        Returns
        -------
        CoMedData
            Self for method chaining
        """
        from .search import search_papers_for_drug_combinations
        
        if not self.drug_combinations:
            self.logger.warning("No drug combinations found. Nothing to search.")
            print("⚠️ No drug combinations found. Nothing to search.")
            return self
        
        print(f"\n{'='*80}\n📊 STEP 1: SEARCHING PUBMED FOR RELEVANT PAPERS\n{'='*80}")
        print(f"• Each drug combination will be searched for up to {retmax} papers")
        print(f"• Results will be saved to {filepath}")
        
        start_time = time.time()
        
        self.papers = search_papers_for_drug_combinations(
            self.drug_combinations, 
            filepath=filepath, 
            verbose=verbose, 
            retmax=retmax, 
            email=email, 
            retry=retry, 
            delay=delay
        )
        
        # Summary after search
        elapsed = time.time() - start_time
        paper_count = len(self.papers)
        print(f"\n✓ Paper search completed in {elapsed:.1f} seconds")
        print(f"✓ Retrieved {paper_count} papers across {len(self.drug_combinations)} drug combinations")
        
        # Show paper count by drug combination
        if paper_count > 0:
            drug_combo_stats = self.papers.groupby(['Drug1', 'Drug2']).size().reset_index(name='count')
            print("\nPapers found per drug combination:")
            for _, row in drug_combo_stats.iterrows():
                print(f"  • {row['Drug1']} + {row['Drug2']}: {row['count']} papers")
        
        return self
        
    def analyze_associations(self, filepath: str = "ddc_papers_association_pd.csv", 
                            verbose: bool = True, max_retries: int = 30, 
                            retry_delay: int = 5) -> 'CoMedData':
        """
        Analyze drug associations from paper abstracts using CoT reasoning.
        
        Parameters
        ----------
        filepath : str, default="ddc_papers_association_pd.csv"
            Path to save association analysis results
        verbose : bool, default=True
            Whether to display progress bar
        max_retries : int, default=30
            Maximum number of retry attempts
        retry_delay : int, default=5
            Delay between retries in seconds
            
        Returns
        -------
        CoMedData
            Self for method chaining
        """
        from .analysis import check_drug_combinations_from_papers, format_reasoning_output
        
        if self.papers.empty:
            self.logger.warning("No papers found. Run search() first.")
            print("⚠️ No papers found. Please run search() first.")
            return self
        
        print(f"\n{'='*80}\n📊 STEP 2: ANALYZING DRUG ASSOCIATIONS IN PAPERS\n{'='*80}")
        print(f"• Using model: {self.config['model_name']}")
        print(f"• Analyzing {len(self.papers)} papers for mentions of drug combinations")
        print(f"• Results will be saved to {filepath}")
        
        start_time = time.time()
        
        self.associations = check_drug_combinations_from_papers(
            self.papers,
            model_name=self.config['model_name'],
            api_key=self.config['api_key'],
            api_base=self.config['api_base'],
            old_openai_api=self.config['old_openai_api'],
            filepath=filepath,
            verbose=verbose,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Format reasoning for better readability
        self.associations = format_reasoning_output(self.associations)
        
        # Summary after association analysis
        elapsed = time.time() - start_time
        assoc_count = len(self.associations)
        positive_count = len(self.associations[self.associations['Combined_medication'].str.lower() == 'yes'])
        
        print(f"\n✓ Association analysis completed in {elapsed:.1f} seconds")
        print(f"✓ Analyzed {assoc_count} papers, found {positive_count} papers with drug combinations")
        
        # Show stats by drug combination
        if positive_count > 0:
            drug_combo_stats = self.associations[self.associations['Combined_medication'].str.lower() == 'yes'].groupby(['Drug1', 'Drug2']).size().reset_index(name='count')
            print("\nPositive associations per drug combination:")
            for _, row in drug_combo_stats.iterrows():
                print(f"  • {row['Drug1']} + {row['Drug2']}: {row['count']} papers")
        
        return self
    
    def analyze_risks(self, filepath: str = "ddc_papers_risk.csv", 
                     verbose: bool = True) -> 'CoMedData':
        """
        Analyze risks from drug combinations using multi-agent system.
        
        Parameters
        ----------
        filepath : str, default="ddc_papers_risk.csv"
            Path to save risk analysis results
        verbose : bool, default=True
            Whether to display progress bar
            
        Returns
        -------
        CoMedData
            Self for method chaining
        """
        from .risk import run_all_risk_evaluations
        
        if self.associations is None or self.associations.empty:
            self.logger.warning("No association analysis found. Run analyze_associations() first.")
            print("⚠️ No association analysis found. Please run analyze_associations() first.")
            return self
        
        # Filter for only 'yes' combined medication
        filtered_associations = self.associations[
            self.associations['Combined_medication'].str.lower() == 'yes'
        ]
        
        if filtered_associations.empty:
            self.logger.warning("No positive drug combinations found.")
            print("⚠️ No positive drug combinations found in the papers.")
            return self
        
        print(f"\n{'='*80}\n📊 STEP 3: ANALYZING RISKS OF DRUG COMBINATIONS\n{'='*80}")
        print(f"• Using model: {self.config['model_name']}")
        print(f"• Analyzing {len(filtered_associations)} papers with confirmed drug combinations")
        print(f"• Evaluating 5 aspects: Risks, Safety, Indications, Selectivity, and Management")
        print(f"• Results will be saved to {filepath}")
        
        start_time = time.time()
        
        self.risk_analysis = run_all_risk_evaluations(
            filtered_associations, 
            model_name=self.config['model_name'],
            api_key=self.config['api_key'],
            api_base=self.config['api_base'],
            old_openai_api=self.config['old_openai_api'],
            verbose=verbose
        )
        
        # Save to file
        if self.risk_analysis is not None and not self.risk_analysis.empty:
            self.risk_analysis.to_csv(filepath, index=False)
            self.logger.info(f"Risk analysis saved to {filepath}")
        
        # Summary after risk analysis
        elapsed = time.time() - start_time
        
        # Calculate statistics for each risk aspect
        aspects = ['Risks', 'Safety', 'Indications', 'Selectivity', 'Management']
        valid_counts = {aspect: (self.risk_analysis[aspect] != 'Invalid').sum() for aspect in aspects}
        
        print(f"\n✓ Risk analysis completed in {elapsed:.1f} seconds")
        print(f"✓ Analyzed {len(self.risk_analysis)} papers for risk factors")
        print("\nValid information found for each aspect:")
        for aspect, count in valid_counts.items():
            print(f"  • {aspect}: {count} papers")
        
        return self
    
    def generate_report(self, output_file: str = "CoMed_Risk_Analysis_Report.html", 
                       verbose: bool = True) -> str:
        """
        Generate HTML report for risk analysis.
        
        Parameters
        ----------
        output_file : str, default="CoMed_Risk_Analysis_Report.html"
            Path to save the HTML report
        verbose : bool, default=True
            Whether to display progress bar
            
        Returns
        -------
        str
            Path to the generated HTML report
        """
        from .report import generate_html_report
        
        if self.risk_analysis is None or self.risk_analysis.empty:
            self.logger.warning("No risk analysis found. Run analyze_risks() first.")
            print("⚠️ No risk analysis found. Please run analyze_risks() first.")
            return None
        
        print(f"\n{'='*80}\n📊 STEP 4: GENERATING HTML REPORT\n{'='*80}")
        print(f"• Using model: {self.config['model_name']}")
        print(f"• Generating comprehensive report for {len(self.risk_analysis)} papers")
        print(f"• Report will be saved to {output_file}")
        
        start_time = time.time()
        
        # Get unique drug combinations
        drug_combos = self.risk_analysis[['Drug1', 'Drug2']].drop_duplicates()
        
        print(f"\nGenerating summaries for {len(drug_combos)} drug combinations:")
        for idx, row in drug_combos.iterrows():
            print(f"  • {row['Drug1']} + {row['Drug2']}")
        
        report_path = generate_html_report(
            self.risk_analysis,
            model_name=self.config['model_name'],
            api_key=self.config['api_key'],
            api_base=self.config['api_base'],
            old_openai_api=self.config['old_openai_api'],
            output_file=output_file,
            verbose=verbose
        )
        
        # Summary after report generation
        elapsed = time.time() - start_time
        print(f"\n✓ Report generation completed in {elapsed:.1f} seconds")
        print(f"✓ Report saved to: {report_path}")
        
        return report_path
    
    def run_full_analysis(self, retmax: int = 30, verbose: bool = True) -> str:
        """
        Run full analysis pipeline from search to report generation.
        
        Parameters
        ----------
        retmax : int, default=30
            Maximum number of records to retrieve per combination
        verbose : bool, default=True
            Whether to display progress bar
            
        Returns
        -------
        str
            Path to the generated HTML report
        """
        total_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🚀 STARTING FULL COMED ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"• Analyzing {len(self.drugs)} drugs with {len(self.drug_combinations)} combinations")
        print(f"• Model: {self.config['model_name']}")
        print(f"• Maximum papers per combination: {retmax}")
        
        # Run each step
        self.search(retmax=retmax, verbose=verbose)
        self.analyze_associations(verbose=verbose)
        self.analyze_risks(verbose=verbose)
        report_path = self.generate_report(verbose=verbose)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print(f"✅ FULL ANALYSIS COMPLETED IN {total_elapsed_time:.1f} SECONDS")
        print(f"{'='*80}")
        print(f"• Papers analyzed: {len(self.papers)}")
        
        if self.associations is not None and not self.associations.empty:
            positive_count = len(self.associations[self.associations['Combined_medication'].str.lower() == 'yes'])
            print(f"• Papers with drug combinations: {positive_count}")
        
        print(f"• Report generated at: {report_path}")
        
        self.logger.info(f"Full analysis completed in {total_elapsed_time:.2f} seconds")
        
        return report_path
    
    def run_ablation_study(self, retmax: int = 30, verbose: bool = True) -> Dict[str, Any]:
        """
        Run ablation study to test independent contributions of each component
        
        Parameters
        ----------
        retmax : int, default=30
            Maximum number of records per combination
        verbose : bool, default=True
            Whether to display progress
            
        Returns
        -------
        Dict[str, Any]
            Ablation study results
        """
        if not self.drug_combinations:
            self.logger.warning("No drug combinations found, cannot perform ablation study")
            print("⚠️ No drug combinations found, cannot perform ablation study")
            return {}
        
        print(f"\n{'='*80}")
        print(f"🔬 Starting Ablation Study")
        print(f"{'='*80}")
        print(f"• Testing {len(self.drug_combinations)} drug combinations")
        print(f"• Model: {self.config['model_name']}")
        
        # Initialize benchmark testing system
        benchmark = CoMedBenchmark(
            model_name=self.config['model_name'],
            api_key=self.config['api_key'],
            api_base=self.config['api_base'],
            old_openai_api=self.config['old_openai_api']
        )
        
        # Run ablation study
        ablation_results = benchmark.run_ablation_study(
            self.drug_combinations, retmax=retmax, verbose=verbose
        )
        
        # Run component ablation study
        component_results = benchmark.run_component_ablation(
            self.drug_combinations, retmax=retmax, verbose=verbose
        )
        
        # Save results
        results_filename = benchmark.save_benchmark_results(ablation_results)
        
        print(f"\n✅ Ablation study completed")
        print(f"📊 Results saved to: {results_filename}")
        
        return {
            "ablation_results": ablation_results,
            "component_results": component_results,
            "results_file": results_filename
        }
    
    def run_component_test(self, component: str, retmax: int = 30, 
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Test individual component performance
        
        Parameters
        ----------
        component : str
            Component to test ('rag', 'cot', 'agents')
        retmax : int, default=30
            Maximum number of records per combination
        verbose : bool, default=True
            Whether to display progress
            
        Returns
        -------
        Dict[str, Any]
            Component test results
        """
        if component not in ['rag', 'cot', 'agents']:
            raise ValueError("Component must be 'rag', 'cot', or 'agents'")
        
        print(f"\n{'='*80}")
        print(f"🧪 Testing Component: {component.upper()}")
        print(f"{'='*80}")
        
        start_time = time.time()
        results = {}
        
        if component == 'rag':
            # Test RAG component
            print("📚 Testing RAG retrieval system...")
            rag_system = RAGSystem()
            rag_results = rag_system.search_drug_combinations(
                self.drug_combinations, retmax=retmax, verbose=verbose
            )
            rag_stats = rag_system.get_retrieval_stats(rag_results)
            
            results = {
                "component": "rag",
                "time": time.time() - start_time,
                "stats": rag_stats,
                "results": rag_results
            }
            
            print(f"✓ RAG test completed in {results['time']:.1f} seconds")
            print(f"  • Papers retrieved: {rag_stats['total_papers']}")
            print(f"  • Drug combinations: {rag_stats['unique_combinations']}")
            
        elif component == 'cot':
            # Test CoT component
            print("🧠 Testing CoT reasoning system...")
            
            # First need RAG results
            if self.papers.empty:
                print("⚠️ No literature data, running RAG retrieval first...")
                self.search(retmax=retmax, verbose=verbose)
            
            cot_reasoner = CoTReasoner(
                model_name=self.config['model_name'],
                api_key=self.config['api_key'],
                api_base=self.config['api_base'],
                old_openai_api=self.config['old_openai_api']
            )
            
            cot_results = cot_reasoner.batch_analyze_associations(
                self.papers, verbose=verbose
            )
            cot_stats = cot_reasoner.get_reasoning_stats(cot_results)
            
            results = {
                "component": "cot",
                "time": time.time() - start_time,
                "stats": cot_stats,
                "results": cot_results
            }
            
            print(f"✓ CoT test completed in {results['time']:.1f} seconds")
            print(f"  • Papers analyzed: {cot_stats['total_papers']}")
            print(f"  • Positive associations: {cot_stats['positive_associations']}")
            print(f"  • Positive rate: {cot_stats['positive_rate']:.2%}")
            
        elif component == 'agents':
            # Test multi-agent component
            print("🤖 Testing multi-agent system...")
            
            # Need CoT results
            if self.associations.empty:
                print("⚠️ No association analysis results, running CoT analysis first...")
                self.analyze_associations(verbose=verbose)
            
            # Filter positive associations
            positive_papers = self.associations[
                self.associations['Combined_medication'].str.lower() == 'yes'
            ]
            
            if positive_papers.empty:
                print("⚠️ No positive association papers, skipping multi-agent test")
                results = {
                    "component": "agents",
                    "time": time.time() - start_time,
                    "stats": {"total_papers": 0},
                    "results": pd.DataFrame()
                }
            else:
                multi_agent_system = MultiAgentSystem(
                    model_name=self.config['model_name'],
                    api_key=self.config['api_key'],
                    api_base=self.config['api_base'],
                    old_openai_api=self.config['old_openai_api']
                )
                
                agent_results = multi_agent_system.batch_process(
                    positive_papers, verbose=verbose
                )
                agent_stats = multi_agent_system.get_agent_stats()
                
                results = {
                    "component": "agents",
                    "time": time.time() - start_time,
                    "stats": agent_stats,
                    "results": agent_results
                }
                
                print(f"✓ Multi-agent test completed in {results['time']:.1f} seconds")
                print(f"  • Papers analyzed: {len(agent_results)}")
                print(f"  • Agents used: {len(agent_stats)}")
        
        return results
    
    def compare_components(self, retmax: int = 30, verbose: bool = True) -> Dict[str, Any]:
        """
        Compare performance of different components
        
        Parameters
        ----------
        retmax : int, default=30
            Maximum number of records per combination
        verbose : bool, default=True
            Whether to display progress
            
        Returns
        -------
        Dict[str, Any]
            Component comparison results
        """
        print(f"\n{'='*80}")
        print(f"📊 Component Performance Comparison")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        # Test each component
        components = ['rag', 'cot', 'agents']
        for component in components:
            print(f"\n🧪 Testing Component: {component.upper()}")
            component_result = self.run_component_test(
                component, retmax=retmax, verbose=verbose
            )
            comparison_results[component] = component_result
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison_results)
        comparison_results["comparison_report"] = comparison_report
        
        print(f"\n{'='*80}")
        print(f"✅ Component comparison completed")
        print(f"{'='*80}")
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate component comparison report
        
        Parameters
        ----------
        comparison_results : Dict[str, Any]
            Component comparison results
            
        Returns
        -------
        Dict[str, Any]
            Comparison report
        """
        report = {
            "performance_summary": {},
            "efficiency_analysis": {},
            "recommendations": []
        }
        
        # Performance summary
        for component, result in comparison_results.items():
            if component == "comparison_report":
                continue
                
            report["performance_summary"][component] = {
                "time": result["time"],
                "papers_processed": result["stats"].get("total_papers", 0),
                "efficiency": result["stats"].get("total_papers", 0) / result["time"] if result["time"] > 0 else 0
            }
        
        # Efficiency analysis
        if "rag" in comparison_results and "cot" in comparison_results:
            rag_time = comparison_results["rag"]["time"]
            cot_time = comparison_results["cot"]["time"]
            
            report["efficiency_analysis"]["cot_overhead"] = cot_time - rag_time
            report["efficiency_analysis"]["cot_contribution"] = (
                comparison_results["cot"]["stats"].get("positive_associations", 0) / 
                comparison_results["cot"]["stats"].get("total_papers", 1)
            )
        
        if "cot" in comparison_results and "agents" in comparison_results:
            cot_time = comparison_results["cot"]["time"]
            agent_time = comparison_results["agents"]["time"]
            
            report["efficiency_analysis"]["agent_overhead"] = agent_time - cot_time
            report["efficiency_analysis"]["agent_contribution"] = (
                len(comparison_results["agents"]["results"]) / 
                comparison_results["agents"]["stats"].get("total_papers", 1)
            )
        
        # Generate recommendations
        if "efficiency_analysis" in report:
            cot_overhead = report["efficiency_analysis"].get("cot_overhead", 0)
            agent_overhead = report["efficiency_analysis"].get("agent_overhead", 0)
            
            if cot_overhead < 30:  # Assume 30 seconds is acceptable
                report["recommendations"].append("CoT reasoning overhead is reasonable, recommend keeping")
            else:
                report["recommendations"].append("CoT reasoning overhead is large, consider optimization")
            
            if agent_overhead < 60:  # Assume 60 seconds is acceptable
                report["recommendations"].append("Multi-agent system overhead is reasonable, recommend keeping")
            else:
                report["recommendations"].append("Multi-agent system overhead is large, consider simplification")
        
        return report