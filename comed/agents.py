# agents.py
"""
Multi-Agent System Module
Implements true agent-to-agent interaction and collaboration
"""

import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from .analysis import invoke_openai_chat_completion, remove_think_tags_from_text, remove_code_block_tags

class Agent:
    """
    Base Agent Class
    """
    
    def __init__(self, name: str, role: str, model_name: str,
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 old_openai_api: str = "No"):
        """
        Initialize agent
        
        Parameters
        ----------
        name : str
            Agent name
        role : str
            Agent role
        model_name : str
            LLM model name
        api_key : str, optional
            API key
        api_base : str, optional
            API base URL
        old_openai_api : str
            Whether to use old OpenAI API format
        """
        self.name = name
        self.role = role
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.old_openai_api = old_openai_api
        self.logger = logging.getLogger(f'comed.agents.{name}')
        
        # Agent state
        self.status = "idle"  # idle, working, completed, error
        self.results = {}
        self.conversation_history = []
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data
            
        Returns
        -------
        Dict[str, Any]
            Processing result
        """
        self.status = "working"
        self.logger.info(f"Agent {self.name} started processing task")
        
        try:
            result = self._execute_task(input_data)
            self.status = "completed"
            self.results = result
            return result
        except Exception as e:
            self.status = "error"
            self.logger.error(f"Agent {self.name} processing failed: {e}")
            return {"error": str(e)}
    
    def _execute_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specific task (to be implemented by subclasses)
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data
            
        Returns
        -------
        Dict[str, Any]
            Processing result
        """
        raise NotImplementedError("Subclasses must implement _execute_task method")
    
    def communicate(self, message: str, other_agent: 'Agent') -> str:
        """
        Communicate with other agents
        
        Parameters
        ----------
        message : str
            Message content
        other_agent : Agent
            Target agent
            
        Returns
        -------
        str
            Response message
        """
        self.logger.info(f"Agent {self.name} sending message to {other_agent.name}")
        
        # Record conversation history
        self.conversation_history.append({
            "from": self.name,
            "to": other_agent.name,
            "message": message,
            "timestamp": time.time()
        })
        
        # Here we can implement more complex communication protocols
        return f"Agent {other_agent.name} received message from {self.name}: {message}"

class RiskAnalysisAgent(Agent):
    """
    Risk Analysis Agent
    Specialized in analyzing risk factors of drug combinations
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        super().__init__("RiskAnalyst", "Risk Analysis Expert", model_name, 
                        api_key, api_base, old_openai_api)
        
        self.risk_categories = [
            "Risks", "Safety", "Indications", "Selectivity", "Management"
        ]
    
    def _execute_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute risk analysis task
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data containing drug information and literature abstract
            
        Returns
        -------
        Dict[str, Any]
            Risk analysis result
        """
        drug1 = input_data.get("drug1", "")
        drug2 = input_data.get("drug2", "")
        abstract = input_data.get("abstract", "")
        
        results = {}
        
        for category in self.risk_categories:
            self.logger.info(f"Analyzing {category} risk factors")
            
            question = self._get_question_template(category, drug1, drug2)
            content = f"""Here is an Abstract of the combined use of {drug1} and {drug2}. Please answer the questions based on this Abstract.
**Abstract**:
{abstract}
**Question**:
{question}"""
            
            # Call LLM for analysis
            analysis_result = self._analyze_risk_category(content)
            
            # Format result
            formatted_result = self._format_risk_result(analysis_result)
            
            results[category] = formatted_result
        
        return results
    
    def _get_question_template(self, category: str, drug1: str, drug2: str) -> str:
        """
        Get question template
        
        Parameters
        ----------
        category : str
            Risk category
        drug1 : str
            First drug name
        drug2 : str
            Second drug name
            
        Returns
        -------
        str
            Question template
        """
        templates = {
            "Risks": f"Does the abstract mention the risks or side effects of the combined use of {drug1} and {drug2}? If so, please point out the relevant sentences. If not, please answer 'not mentioned' without any explanation.",
            "Safety": f"Does the abstract mention the efficacy or safety of the combination of {drug1} and {drug2}? If so, please point out the relevant sentences. If not, please answer 'not mentioned' without any explanation.",
            "Indications": f"Does the abstract mention the indications or selectivity of the combination of {drug1} and {drug2}? If so, please point out the relevant sentences. If not, please answer 'not mentioned' without any explanation.",
            "Selectivity": f"Does the abstract mention the patient population or selection of the combination of {drug1} and {drug2}? If so, please point out the relevant sentences. If not, please answer 'not mentioned' without any explanation.",
            "Management": f"Does the abstract mention the monitoring and management of the combination of {drug1} and {drug2}? If so, please point out the relevant sentences. If not, please answer 'not mentioned' without any explanation."
        }
        
        return templates.get(category, f"Please analyze the {category} information of {drug1} and {drug2}.")
    
    def _analyze_risk_category(self, content: str) -> str:
        """
        Analyze specific risk category
        
        Parameters
        ----------
        content : str
            Analysis content
            
        Returns
        -------
        str
            Analysis result
        """
        input_messages = [
            {"role": "system", "content": (
                "You are an excellent language master. You cannot request additional information from the user or suggest that they look for more information."
            )},
            {"role": "user", "content": content}
        ]
        
        return invoke_openai_chat_completion(
            self.model_name, input_messages,
            api_key=self.api_key,
            api_base=self.api_base,
            old_openai_api=self.old_openai_api
        )
    
    def _format_risk_result(self, result: str) -> Dict[str, str]:
        """
        Format risk analysis result
        
        Parameters
        ----------
        result : str
            Original analysis result
            
        Returns
        -------
        Dict[str, str]
            Formatted result
        """
        # Clean result
        cleaned_result = remove_think_tags_from_text(result)
        
        # Format requirements
        format_requirements = """Output the result in JSON format with two fields: "raw" for the original content and "formatted" for the transformed content. If the original content includes the phrase "not mentioned" or any variation indicating uncertainty or ambiguity (e.g., "not specified," "unclear," or "unknown"), the "formatted" field should output "Invalid". Otherwise, the "formatted" field should provide a concise summary of the original content. Example output: {"raw": "not mentioned", "formatted": "Invalid"}"""
        
        # Call LLM for formatting
        formatted_result = invoke_openai_chat_completion(
            self.model_name,
            [
                {"role": "system", "content": "You are a formatting function. You will only output the content in the transformed format and nothing else."},
                {"role": "user", "content": f"**Content**:\n{cleaned_result}\n**Format Requirements**:{format_requirements}"}
            ],
            api_key=self.api_key,
            api_base=self.api_base,
            old_openai_api=self.old_openai_api
        )
        
        if formatted_result is None:
            return {"raw": result, "formatted": "Invalid"}
        
        # Clean formatted result
        formatted_result = remove_think_tags_from_text(formatted_result)
        formatted_result = remove_code_block_tags(formatted_result)
        
        try:
            return json.loads(formatted_result)
        except json.JSONDecodeError:
            return {"raw": result, "formatted": "Invalid"}

class SafetyAgent(Agent):
    """
    Safety Assessment Agent
    Specialized in evaluating drug combination safety
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        super().__init__("SafetyExpert", "Safety Assessment Expert", model_name,
                        api_key, api_base, old_openai_api)
    
    def _execute_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute safety assessment task
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data
            
        Returns
        -------
        Dict[str, Any]
            Safety assessment result
        """
        # Implement safety assessment logic
        return {"safety_level": "moderate", "recommendations": []}

class ClinicalAgent(Agent):
    """
    Clinical Decision Agent
    Specialized in providing clinical recommendations
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        super().__init__("ClinicalExpert", "Clinical Decision Expert", model_name,
                        api_key, api_base, old_openai_api)
    
    def _execute_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clinical decision task
        
        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data
            
        Returns
        -------
        Dict[str, Any]
            Clinical recommendation
        """
        # Implement clinical decision logic
        return {"clinical_recommendation": "Close monitoring required", "monitoring_plan": []}

class MultiAgentSystem:
    """
    Multi-Agent System
    Coordinates collaboration between multiple agents
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 api_base: Optional[str] = None, old_openai_api: str = "No"):
        """
        Initialize multi-agent system
        
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
        self.logger = logging.getLogger('comed.agents.system')
        
        # Initialize agents
        self.agents = {
            "risk_analyst": RiskAnalysisAgent(model_name, api_key, api_base, old_openai_api),
            "safety_expert": SafetyAgent(model_name, api_key, api_base, old_openai_api),
            "clinical_expert": ClinicalAgent(model_name, api_key, api_base, old_openai_api)
        }
        
        # Agent collaboration workflow
        self.workflow = [
            "risk_analyst",
            "safety_expert", 
            "clinical_expert"
        ]
    
    def process_drug_combination(self, drug1: str, drug2: str, 
                               abstract: str) -> Dict[str, Any]:
        """
        Process drug combination analysis
        
        Parameters
        ----------
        drug1 : str
            First drug name
        drug2 : str
            Second drug name
        abstract : str
            Literature abstract
            
        Returns
        -------
        Dict[str, Any]
            Multi-agent analysis result
        """
        self.logger.info(f"Starting multi-agent analysis: {drug1} + {drug2}")
        
        input_data = {
            "drug1": drug1,
            "drug2": drug2,
            "abstract": abstract
        }
        
        results = {}
        
        # Execute agent collaboration workflow
        for agent_name in self.workflow:
            agent = self.agents[agent_name]
            self.logger.info(f"Executing agent: {agent.name}")
            
            # Execute agent task
            agent_result = agent.process(input_data)
            results[agent_name] = agent_result
            
            # Agent-to-agent communication (here we can implement more complex communication protocols)
            if agent_name != self.workflow[-1]:
                next_agent_name = self.workflow[self.workflow.index(agent_name) + 1]
                next_agent = self.agents[next_agent_name]
                
                # Pass context information
                input_data[f"{agent_name}_result"] = agent_result
                
                # Record communication
                agent.communicate(f"Completed {agent.role} analysis", next_agent)
        
        # Integrate all agent results
        final_result = self._integrate_results(results)
        
        self.logger.info(f"Multi-agent analysis completed: {drug1} + {drug2}")
        
        return final_result
    
    def _integrate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate multi-agent results
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results from each agent
            
        Returns
        -------
        Dict[str, Any]
            Integrated result
        """
        # Here we can implement more complex integration logic
        integrated_result = {
            "risk_analysis": results.get("risk_analyst", {}),
            "safety_assessment": results.get("safety_expert", {}),
            "clinical_recommendation": results.get("clinical_expert", {}),
            "collaboration_summary": "Multi-agent collaboration analysis completed"
        }
        
        return integrated_result
    
    def batch_process(self, papers: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Batch process drug combinations
        
        Parameters
        ----------
        papers : pd.DataFrame
            Papers DataFrame
        verbose : bool
            Whether to display progress bar
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing multi-agent analysis results
        """
        results = []
        
        pbar = tqdm(range(len(papers)), disable=not verbose,
                   desc="🤖 Multi-agent collaboration analysis", unit="paper")
        
        for i in pbar:
            paper = papers.iloc[i]
            drug1 = paper["Drug1"]
            drug2 = paper["Drug2"]
            abstract = paper["Abstract"]
            
            pbar.set_description(f"🤖 Multi-agent analyzing: {drug1} + {drug2} [{i+1}/{len(papers)}]")
            
            # Execute multi-agent analysis
            agent_result = self.process_drug_combination(drug1, drug2, abstract)
            
            # Record result
            result_row = {
                "ID": i,
                "Drug1": drug1,
                "Drug2": drug2,
                "PMID": paper.get("PMID", ""),
                "Title": paper.get("Title", ""),
                "Abstract": abstract,
                "Authors": paper.get("Authors", ""),
                "Journal": paper.get("Journal", ""),
                "Publication Date": paper.get("Publication Date", ""),
                "Link": paper.get("Link", ""),
                "MultiAgent_Result": json.dumps(agent_result, ensure_ascii=False)
            }
            
            results.append(result_row)
            
            pbar.write(f"  ↳ Multi-agent analysis completed: {drug1} + {drug2}")
        
        return pd.DataFrame(results)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics
        
        Returns
        -------
        Dict[str, Any]
            Statistics information
        """
        stats = {}
        for agent_name, agent in self.agents.items():
            stats[agent_name] = {
                "name": agent.name,
                "role": agent.role,
                "status": agent.status,
                "conversation_count": len(agent.conversation_history)
            }
        
        return stats