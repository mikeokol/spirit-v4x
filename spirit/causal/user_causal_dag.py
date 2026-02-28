import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

from spirit.models.mechanisms import MechanismActivation

@dataclass
class CausalNode:
    id: str
    label: str
    category: str  # 'biological', 'cognitive', 'emotional', 'behavior'
    current_value: float
    stability: float

@dataclass
class CausalEdge:
    source: str
    target: str
    strength: float  # -1 to 1
    mechanism: str
    confidence: float

class UserCausalDAG:
    """
    Living representation of user's behavioral causal structure
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.nodes = {}
        self.edges = []
        self._init_default_graph()
    
    def _init_default_graph(self):
        """Initialize with standard nodes"""
        self.nodes = {
            'sleep_debt': CausalNode('sleep_debt', 'Sleep Debt', 'biological', 0.3, 0.9),
            'cognitive_energy': CausalNode('cognitive_energy', 'Energy', 'biological', 0.6, 0.85),
            'stress': CausalNode('stress', 'Stress', 'emotional', 0.4, 0.8),
            'ambiguity': CausalNode('ambiguity', 'Ambiguity', 'cognitive', 0.7, 0.75),
            'identity_threat': CausalNode('identity_threat', 'Identity Threat', 'social', 0.2, 0.6),
            'procrastination': CausalNode('procrastination', 'Procrastination', 'behavior', 0.8, 0.9),
            'deep_work': CausalNode('deep_work', 'Deep Work', 'outcome', 0.3, 0.8),
        }
        
        self.edges = [
            CausalEdge('sleep_debt', 'cognitive_energy', -0.7, 
                      MechanismActivation.SLEEP_DEBT_IMPAIRMENT.name, 0.9),
            CausalEdge('cognitive_energy', 'deep_work', 0.6, 
                      MechanismActivation.COGNITIVE_ENERGY_DEPLETION.name, 0.85),
            CausalEdge('stress', 'procrastination', 0.5, 
                      MechanismActivation.AVOIDANCE_REINFORCEMENT.name, 0.75),
            CausalEdge('ambiguity', 'procrastination', 0.8, 
                      MechanismActivation.AMBIGUITY_COST.name, 0.8),
            CausalEdge('identity_threat', 'procrastination', 0.6, 
                      MechanismActivation.IDENTITY_DEFENSIVE_REASONING.name, 0.7),
        ]
    
    def get_primary_drivers(self, target: str = 'procrastination') -> List[Tuple[str, float]]:
        """Ranked list of what drives a behavior"""
        drivers = []
        for edge in self.edges:
            if edge.target == target:
                drivers.append((edge.source, abs(edge.strength) * edge.confidence))
        return sorted(drivers, key=lambda x: x[1], reverse=True)
    
    def generate_insight_summary(self) -> str:
        """Natural language explanation of causal structure"""
        drivers = self.get_primary_drivers('procrastination')
        
        if not drivers:
            return "Insufficient data for causal analysis."
        
        summary = f"""Your Behavioral Causal Map (Auto-Generated)
        
Primary Drivers of Procrastination:
1. {self.nodes[drivers[0][0]].label} ({drivers[0][1]:.0%} influence)
   → {self._get_mechanism_explanation(drivers[0][0])}
   
2. {self.nodes[drivers[1][0]].label} ({drivers[1][1]:.0%} influence)
   → {self._get_mechanism_explanation(drivers[1][0])}

Key Insight: Your procrastination is primarily driven by {self.nodes[drivers[0][0]].label.lower()}, not willpower.
Recommendation: Spirit suggests targeting {self.nodes[drivers[0][0]].label.lower()} first.
"""
        return summary
    
    def _get_mechanism_explanation(self, node_id: str) -> str:
        explanations = {
            'ambiguity': "You avoid work most when tasks feel undefined",
            'identity_threat': "When you feel 'not cut out for this,' you protect yourself by not trying",
            'stress': "High stress triggers threat-avoidance mode",
            'sleep_debt': "Poor sleep impairs inhibition and planning"
        }
        return explanations.get(node_id, "Complex interaction of factors")
    
    def export_to_json(self) -> Dict:
        """Export for D3.js visualization"""
        return {
            'nodes': [
                {
                    'id': n.id,
                    'label': n.label,
                    'category': n.category,
                    'value': n.current_value,
                    'stability': n.stability
                } for n in self.nodes.values()
            ],
            'links': [
                {
                    'source': e.source,
                    'target': e.target,
                    'value': abs(e.strength),
                    'type': 'inhibits' if e.strength < 0 else 'promotes',
                    'mechanism': e.mechanism,
                    'confidence': e.confidence
                } for e in self.edges
            ]
        }
    
    def add_discovered_edge(self, source: str, target: str, strength: float, 
                           mechanism: MechanismActivation, confidence: float):
        """Add new causal link discovered by Digital Twin"""
        self.edges.append(CausalEdge(
            source=source, target=target, strength=strength,
            mechanism=mechanism.name, confidence=confidence
        ))
