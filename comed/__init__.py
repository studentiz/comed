"""
CoMed: A framework for analyzing co-medication risks using Chain-of-Thought reasoning.
Modular architecture supporting RAG, CoT, and multi-agent systems with ablation study capabilities.

⚠️  IMPORTANT DISCLAIMER:
This tool is designed for RESEARCH AND EDUCATIONAL PURPOSES ONLY.
NOT intended for direct clinical use or automated clinical decision-making.
Always consult qualified healthcare professionals for clinical decisions.
"""

from .core import CoMedData
from .rag import RAGSystem
from .cot import CoTReasoner
from .agents import MultiAgentSystem, Agent, RiskAnalysisAgent, SafetyAgent, ClinicalAgent
from .benchmark import CoMedBenchmark
from . import utils
from . import io

__version__ = "2.0.1"
__all__ = [
    'CoMedData',
    'RAGSystem', 
    'CoTReasoner',
    'MultiAgentSystem',
    'Agent',
    'RiskAnalysisAgent',
    'SafetyAgent', 
    'ClinicalAgent',
    'CoMedBenchmark',
    'utils',
    'io'
]