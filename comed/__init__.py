"""
CoMed: A framework for analyzing co-medication risks using Chain-of-Thought reasoning.
Modular architecture supporting RAG, CoT, and multi-agent systems.

⚠️  IMPORTANT DISCLAIMER:
This tool is designed for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
NOT intended for direct clinical use or automated clinical decision-making.
Always consult qualified healthcare professionals for clinical decisions.
"""

from .core import CoMedData
from .rag import RAGSystem
from .cot import CoTReasoner
from .agents import MultiAgentSystem, Agent, RiskAnalysisAgent, SafetyAgent, ClinicalAgent, ConflictResolver, CollectiveDecisionMaker
from . import utils
from . import io

__version__ = "2.1.1"
__all__ = [
    'CoMedData',
    'RAGSystem', 
    'CoTReasoner',
    'MultiAgentSystem',
    'Agent',
    'RiskAnalysisAgent',
    'SafetyAgent', 
    'ClinicalAgent',
    'ConflictResolver',
    'CollectiveDecisionMaker',
    'utils',
    'io'
]