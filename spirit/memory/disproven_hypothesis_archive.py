"""
Disproven Hypothesis Archive (DHA): The anti-pattern database.
Stores what almost seemed true but wasn't.

Prevents personality locking by tracking failed hypotheses across life phases.
Enables faster learning from "what didn't work" than from confirmations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
import json
import hashlib

from spirit.db.supabase_client import get_behavioral_store
from spirit.evidence.personal_evidence_ladder import EvidenceLevel


class FalsificationType(Enum):
    """How the hypothesis was proven wrong."""
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"  # Data showed opposite
    BOUNDARY_CONDITION_FAILURE = "boundary_condition_failure"  # Worked in some contexts, not others
    TEMPORAL_DECAY = "temporal_decay"  # Was true, stopped being true (life phase change)
    INTERVENTION_REVERSAL = "intervention_reversal"  # Manipulation produced opposite effect
    CONFOUND_DISCOVERY = "confound_discovery"  # Hidden variable explained the pattern
    BASELINE_REGRESSION = "baseline_regression"  # Just regression to the mean
    USER_EXPLICIT_REJECTION = "user_explicit_rejection"  # User said "that's not me"


@dataclass
class DisprovenHypothesis:
    """
    A hypothesis that was once entertained but later falsified.
    Rich metadata to prevent re-testing and enable learning.
    """
    hypothesis_id: str
    user_id: str
    
    # The hypothesis (what was believed)
    original_hypothesis: str
    hypothesis_type: str  # 'causal', 'correlational', 'trait', 'preference'
    
    # How it was formed
    evidence_level_at_formation: EvidenceLevel
    formed_at: datetime
    supporting_evidence_count: int
    
    # How it died
    falsified_at: datetime
    falsification_type: FalsificationType
    falsification_evidence: Dict[str, Any]  # The data that killed it
    
    # Context of failure
    life_phase_context: Optional[str]  # "new_job", "post_breakup", "pandemic", etc.
    confounds_present: List[str]  # What hidden variables were involved
    
    # What replaced it (if anything)
    successor_hypothesis: Optional[str]  # More nuanced replacement
    nuance_added: Optional[str]  # "Only true on weekdays", etc.
    
    # Metadata for learning
    time_to_falsification_days: float
    confidence_at_peak: float
    confidence_at_death: float
    
    # Search/retrieval
    keyword_fingerprint: Set[str]  # For similarity matching
    related_hypotheses: List[str]  # Other hypotheses falsified around same time
    
    # Archival status
    archive_status: str  # 'active', 'superseded', 'revived', 'permanently_false'


class DisprovenHypothesisArchive:
    """
    Manages the archive of falsified hypotheses.
    Queryable by the BehavioralScientistAgent to prevent re-testing.
    """
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.temporal_relevance_decay_days = 90  # Older falsifications less relevant
        self.life_phase_buffer_days = 30  # Grace period for life phase changes
    
    async def archive_hypothesis(
        self,
        user_id: str,
        original_hypothesis: str,
        hypothesis_type: str,
        evidence_level: EvidenceLevel,
        falsification_type: FalsificationType,
        falsification_evidence: Dict[str, Any],
        successor_hypothesis: Optional[str] = None,
        nuance_added: Optional[str] = None
    ) -> DisprovenHypothesis:
        """
        Archive a falsified hypothesis with full context.
        """
        # Generate fingerprint for similarity matching
        fingerprint = self._generate_fingerprint(original_hypothesis)
        
        # Calculate time to falsification
        formed_at = falsification_evidence.get('hypothesis_formed_at', datetime.utcnow().isoformat())
        if isinstance(formed_at, str):
            formed_at = datetime.fromisoformat(formed_at.replace('Z', '+00:00'))
        
        time_to_falsify = (datetime.utcnow() - formed_at).total_seconds() / 86400
        
        # Detect life phase context
        life_phase = await self._detect_life_phase_context(user_id)
        
        # Get related hypotheses (falsified within same window)
        related = await self._find_related_falsifications(user_id, formed_at)
        
        disproven = DisprovenHypothesis(
            hypothesis_id=f"falsified_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            original_hypothesis=original_hypothesis,
            hypothesis_type=hypothesis_type,
            evidence_level_at_formation=evidence_level,
            formed_at=formed_at,
            supporting_evidence_count=falsification_evidence.get('supporting_count', 0),
            falsified_at=datetime.utcnow(),
            falsification_type=falsification_type,
            falsification_evidence=falsification_evidence,
            life_phase_context=life_phase,
            confounds_present=falsification_evidence.get('confounds_identified', []),
            successor_hypothesis=successor_hypothesis,
            nuance_added=nuance_added,
            time_to_falsification_days=time_to_falsify,
            confidence_at_peak=falsification_evidence.get('peak_confidence', 0.8),
            confidence_at_death=falsification_evidence.get('death_confidence', 0.2),
            keyword_fingerprint=fingerprint,
            related_hypotheses=[r['hypothesis_id'] for r in related],
            archive_status='active'
        )
        
        # Persist
        await self._persist_hypothesis(disproven)
        
        # Update collective intelligence (anonymized)
        await self._update_collective_falsification_patterns(disproven)
        
        return disproven
    
    async def check_similar_falsifications(
        self,
        user_id: str,
        proposed_hypothesis: str,
        current_life_phase: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Check if similar hypothesis was already falsified.
        Returns ranked list of similar disproven hypotheses with relevance scores.
        """
        store = await get_behavioral_store()
        if not store:
            return []
        
        # Get user's archive
        archive = store.client.table('disproven_hypotheses').select('*').eq(
            'user_id', user_id
        ).eq('archive_status', 'active').execute()
        
        if not archive.data:
            return []
        
        proposed_fingerprint = self._generate_fingerprint(proposed_hypothesis)
        
        matches = []
        for record in archive.data:
            # Calculate similarity
            archived_fingerprint = set(record.get('keyword_fingerprint', []))
            if not archived_fingerprint:
                continue
            
            similarity = self._calculate_similarity(proposed_fingerprint, archived_fingerprint)
            
            if similarity < self.similarity_threshold:
                continue
            
            # Calculate temporal relevance
            falsified_at = datetime.fromisoformat(record['falsified_at'].replace('Z', '+00:00'))
            days_since = (datetime.utcnow() - falsified_at).days
            temporal_relevance = max(0, 1 - (days_since / self.temporal_relevance_decay_days))
            
            # Check life phase relevance
            phase_match = 1.0
            if current_life_phase and record.get('life_phase_context'):
                if current_life_phase == record['life_phase_context']:
                    phase_match = 1.0  # Same phase - highly relevant warning
                else:
                    phase_match = 0.3  # Different phase - might be valid now
            
            # Combined relevance
            relevance = (similarity * 0.5) + (temporal_relevance * 0.3) + (phase_match * 0.2)
            
            matches.append({
                'hypothesis_id': record['hypothesis_id'],
                'original_hypothesis': record['original_hypothesis'],
                'similarity': similarity,
                'temporal_relevance': temporal_relevance,
                'phase_match': phase_match,
                'combined_relevance': relevance,
                'falsification_type': record['falsification_type'],
                'falsified_at': record['falsified_at'],
                'life_phase_context': record.get('life_phase_context'),
                'nuance_added': record.get('nuance_added'),
                'successor_hypothesis': record.get('successor_hypothesis'),
                'warning': self._generate_warning(record, similarity, phase_match)
            })
        
        # Sort by relevance
        matches.sort(key=lambda x: x['combined_relevance'], reverse=True)
        
        return matches[:5]  # Top 5 most relevant
    
    async def get_learning_summary(
        self,
        user_id: str,
        timeframe_days: int = 90
    ) -> Dict[str, Any]:
        """
        Generate summary of what Spirit has learned from falsifications.
        Shows evolution of understanding.
        """
        store = await get_behavioral_store()
        if not store:
            return {"error": "no_data_store"}
        
        since = (datetime.utcnow() - timedelta(days=timeframe_days)).isoformat()
        
        archive = store.client.table('disproven_hypotheses').select('*').eq(
            'user_id', user_id
        ).gte('falsified_at', since).execute()
        
        if not archive.data:
            return {
                "period_days": timeframe_days,
                "falsifications_count": 0,
                "message": "No hypotheses falsified in this period - may indicate insufficient testing"
            }
        
        # Analyze patterns
        by_type = {}
        by_falsification_method = {}
        avg_time_to_falsify = []
        life_phases = set()
        
        for record in archive.data:
            h_type = record['hypothesis_type']
            f_type = record['falsification_type']
            
            by_type[h_type] = by_type.get(h_type, 0) + 1
            by_falsification_method[f_type] = by_falsification_method.get(f_type, 0) + 1
            
            avg_time_to_falsify.append(record.get('time_to_falsification_days', 0))
            
            if record.get('life_phase_context'):
                life_phases.add(record['life_phase_context'])
        
        # Calculate learning velocity
        learning_velocity = len(archive.data) / timeframe_days
        
        # Identify recurring failure modes
        recurring_failures = [
            ft for ft, count in by_falsification_method.items()
            if count > len(archive.data) * 0.3
        ]
        
        return {
            "period_days": timeframe_days,
            "falsifications_count": len(archive.data),
            "learning_velocity_per_week": round(learning_velocity * 7, 2),
            "hypothesis_types_tested": by_type,
            "falsification_methods": by_falsification_method,
            "avg_time_to_falsification_days": round(sum(avg_time_to_falsify) / len(avg_time_to_falsify), 1) if avg_time_to_falsify else 0,
            "life_phases_detected": list(life_phases),
            "recurring_failure_modes": recurring_failures,
            "key_insights": self._generate_insights(archive.data),
            "current_understanding_confidence": self._calculate_current_confidence(archive.data)
        }
    
    async def attempt_revival(
        self,
        hypothesis_id: str,
        new_evidence: Dict[str, Any],
        revival_reason: str
    ) -> Dict[str, Any]:
        """
        Attempt to revive a falsified hypothesis with new evidence.
        Strict criteria - most revivals should fail.
        """
        store = await get_behavioral_store()
        if not store:
            return {"error": "no_data_store"}
        
        # Get archived hypothesis
        record = store.client.table('disproven_hypotheses').select('*').eq(
            'hypothesis_id', hypothesis_id
        ).execute()
        
        if not record.data:
            return {"error": "hypothesis_not_found"}
        
        archived = record.data[0]
        
        # Check if enough time has passed (life phase change likely)
        falsified_at = datetime.fromisoformat(archived['falsified_at'].replace('Z', '+00:00'))
        days_since = (datetime.utcnow() - falsified_at).days
        
        if days_since < self.temporal_relevance_decay_days:
            return {
                "revival_allowed": False,
                "reason": f"Only {days_since} days since falsification. Wait at least {self.temporal_relevance_decay_days} days.",
                "suggested_action": "design_experiment_to_test_under_new_conditions"
            }
        
        # Check if life phase changed
        current_phase = await self._detect_life_phase_context(archived['user_id'])
        original_phase = archived.get('life_phase_context')
        
        if current_phase == original_phase:
            return {
                "revival_allowed": False,
                "reason": "Life phase unchanged. Original falsification likely still valid.",
                "suggested_action": "formulate_nuanced_variant_instead"
            }
        
        # Check new evidence strength
        evidence_level = new_evidence.get('evidence_level', 0)
        if evidence_level < EvidenceLevel.COUNTERFACTUAL_STABILITY.value:
            return {
                "revival_allowed": False,
                "reason": f"New evidence level {evidence_level} insufficient. Need Level 4+.",
                "suggested_action": "accumulate_stronger_evidence"
            }
        
        # Allow revival with strict tracking
        store.client.table('disproven_hypotheses').update({
            'archive_status': 'revived',
            'revived_at': datetime.utcnow().isoformat(),
            'revival_reason': revival_reason,
            'revival_evidence': new_evidence
        }).eq('hypothesis_id', hypothesis_id).execute()
        
        return {
            "revival_allowed": True,
            "message": f"Hypothesis revived after {days_since} days due to life phase change ({original_phase} → {current_phase})",
            "warnings": [
                "Monitor closely for rapid re-falsification",
                "If falsified again within 30 days, permanently archive",
                "Consider more nuanced formulation"
            ]
        }
    
    async def get_falsification_patterns_for_archetype(
        self,
        archetype_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get common falsification patterns for a behavioral archetype.
        Powers: "People like you often mistakenly believe..."
        """
        store = await get_behavioral_store()
        if not store:
            return []
        
        # Get anonymized patterns for archetype
        patterns = store.client.table('collective_falsification_patterns').select('*').eq(
            'archetype_name', archetype_name
        ).order('frequency', desc=True).limit(10).execute()
        
        if not patterns.data:
            return []
        
        return [
            {
                "common_misconception": p['hypothesis_pattern'],
                "how_it_usually_fails": p['common_falsification_type'],
                "frequency_in_archetype": p['frequency'],
                "typical_nuance": p['typical_replacement'],
                "warning_signs": p.get('early_warning_indicators', [])
            }
            for p in patterns.data
        ]
    
    # Helper methods
    
    def _generate_fingerprint(self, hypothesis_text: str) -> Set[str]:
        """Generate keyword fingerprint for similarity matching."""
        # Simple keyword extraction - could use NLP
        text = hypothesis_text.lower()
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'and', 'but', 'or', 'yet', 'so', 'if',
                     'because', 'although', 'though', 'while', 'where', 'when',
                     'that', 'which', 'who', 'whom', 'whose', 'what', 'this',
                     'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                     'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
                     'its', 'our', 'their', 'user', 'tends', 'tend', 'often',
                     'sometimes', 'usually', 'always', 'never', 'user'}
        
        words = [w.strip('.,;:!?()[]{}') for w in text.split()]
        keywords = {w for w in words if len(w) > 3 and w not in stopwords}
        
        return keywords
    
    def _calculate_similarity(self, fp1: Set[str], fp2: Set[str]) -> float:
        """Calculate Jaccard similarity between fingerprints."""
        if not fp1 or not fp2:
            return 0.0
        
        intersection = len(fp1 & fp2)
        union = len(fp1 | fp2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _detect_life_phase_context(self, user_id: str) -> Optional[str]:
        """Detect current life phase from recent data."""
        store = await get_behavioral_store()
        if not store:
            return None
        
        # Check for major changes in patterns
        recent = store.client.table('behavioral_observations').select('*').eq(
            'user_id', user_id
        ).order('timestamp', desc=True).limit(100).execute()
        
        if not recent.data:
            return None
        
        # Simple heuristics - could be more sophisticated
        recent_apps = set()
        recent_hours = []
        
        for obs in recent.data:
            behavior = obs.get('behavior', {})
            recent_apps.add(behavior.get('app_category', 'unknown'))
            
            ts = datetime.fromisoformat(obs['timestamp'].replace('Z', '+00:00'))
            recent_hours.append(ts.hour)
        
        # Detect "new job" (productivity apps + regular hours)
        if 'productivity' in recent_apps and recent_hours:
            hour_variance = max(recent_hours) - min(recent_hours)
            if hour_variance < 6:  # Regular schedule
                return "structured_work_period"
        
        # Detect "transition" (high variance, exploration)
        if len(recent_apps) > 5:
            return "exploration_phase"
        
        return "stable_period"
    
    async def _find_related_falsifications(
        self,
        user_id: str,
        around_time: datetime,
        window_days: int = 7
    ) -> List[Dict]:
        """Find other hypotheses falsified around the same time."""
        store = await get_behavioral_store()
        if not store:
            return []
        
        start = (around_time - timedelta(days=window_days)).isoformat()
        end = (around_time + timedelta(days=window_days)).isoformat()
        
        related = store.client.table('disproven_hypotheses').select('*').eq(
            'user_id', user_id
        ).gte('falsified_at', start).lte('falsified_at', end).execute()
        
        return related.data if related.data else []
    
    def _generate_warning(
        self,
        archived: Dict,
        similarity: float,
        phase_match: float
    ) -> str:
        """Generate human-readable warning about similar falsified hypothesis."""
        hypothesis = archived['original_hypothesis']
        falsification = archived['falsification_type'].replace('_', ' ')
        
        if phase_match > 0.8:
            return (
                f"⚠️ You previously believed '{hypothesis[:60]}...' "
                f"but this was proven wrong by {falsification}. "
                f"Be cautious about reaching the same conclusion again."
            )
        else:
            return (
                f"ℹ️ In a different life phase, you believed '{hypothesis[:60]}...' "
                f"which turned out to be wrong. This may or may not apply now."
            )
    
    def _generate_insights(self, archive_data: List[Dict]) -> List[str]:
        """Generate key insights from falsification patterns."""
        insights = []
        
        # Check for overconfidence in early hypotheses
        quick_falsifications = [
            a for a in archive_data 
            if a.get('time_to_falsification_days', 999) < 3
        ]
        if len(quick_falsifications) > len(archive_data) * 0.3:
            insights.append(
                "You tend to form hypotheses quickly that don't hold up. "
                "Consider longer observation periods before concluding."
            )
        
        # Check for temporal decay patterns
        temporal_decays = [
            a for a in archive_data 
            if a['falsification_type'] == FalsificationType.TEMPORAL_DECAY.value
        ]
        if len(temporal_decays) > 3:
            insights.append(
                "Many of your patterns change over time. What was true 3 months "
                "ago may not be true now. Spirit will weight recent evidence higher."
            )
        
        # Check for confound sensitivity
        confound_fails = [
            a for a in archive_data
            if a['falsification_type'] == FalsificationType.CONFOUND_DISCOVERY.value
        ]
        if len(confound_fails) > 2:
            insights.append(
                "Your behavior is sensitive to hidden variables (sleep, stress, etc.). "
                "Spirit will be more rigorous about confound detection for you."
            )
        
        return insights
    
    def _calculate_current_confidence(self, archive_data: List[Dict]) -> float:
        """Calculate current confidence in Spirit's understanding."""
        if not archive_data:
            return 0.5  # Neutral - not enough testing
        
        # More falsifications with faster learning = higher confidence
        # Paradoxically, knowing what you're wrong about increases confidence
        
        recent = [
            a for a in archive_data 
            if (datetime.utcnow() - datetime.fromisoformat(a['falsified_at'].replace('Z', '+00:00'))).days < 30
        ]
        
        if len(recent) > 5:
            # Rapid falsification suggests active learning but volatile understanding
            return 0.6
        
        if len(archive_data) > 10 and len(recent) < 3:
            # Many historical falsifications, stable recently = mature understanding
            return 0.85
        
        return 0.7
    
    async def _persist_hypothesis(self, hypothesis: DisprovenHypothesis):
        """Save to database."""
        store = await get_behavioral_store()
        if not store:
            return
        
        store.client.table('disproven_hypotheses').insert({
            'hypothesis_id': hypothesis.hypothesis_id,
            'user_id': hypothesis.user_id,
            'original_hypothesis': hypothesis.original_hypothesis,
            'hypothesis_type': hypothesis.hypothesis_type,
            'evidence_level_at_formation': hypothesis.evidence_level_at_formation.value,
            'formed_at': hypothesis.formed_at.isoformat(),
            'supporting_evidence_count': hypothesis.supporting_evidence_count,
            'falsified_at': hypothesis.falsified_at.isoformat(),
            'falsification_type': hypothesis.falsification_type.value,
            'falsification_evidence': hypothesis.falsification_evidence,
            'life_phase_context': hypothesis.life_phase_context,
            'confounds_present': hypothesis.confounds_present,
            'successor_hypothesis': hypothesis.successor_hypothesis,
            'nuance_added': hypothesis.nuance_added,
            'time_to_falsification_days': hypothesis.time_to_falsification_days,
            'confidence_at_peak': hypothesis.confidence_at_peak,
            'confidence_at_death': hypothesis.confidence_at_death,
            'keyword_fingerprint': list(hypothesis.keyword_fingerprint),
            'related_hypotheses': hypothesis.related_hypotheses,
            'archive_status': hypothesis.archive_status
        }).execute()
    
    async def _update_collective_intelligence(self, hypothesis: DisprovenHypothesis):
        """Update anonymized collective patterns."""
        store = await get_behavioral_store()
        if not store:
            return
        
        # Get user archetype
        from spirit.memory.collective_intelligence import CollectiveIntelligenceEngine
        collective = CollectiveIntelligenceEngine()
        archetype = await collective.get_user_archetype(hypothesis.user_id)
        
        if not archetype:
            return
        
        # Update or create pattern record
        existing = store.client.table('collective_falsification_patterns').select('*').eq(
            'archetype_name', archetype.name
        ).eq('hypothesis_pattern', hypothesis.hypothesis_type).execute()
        
        if existing.data:
            # Update frequency
            current_freq = existing.data[0].get('frequency', 0)
            store.client.table('collective_falsification_patterns').update({
                'frequency': current_freq + 1,
                'last_updated': datetime.utcnow().isoformat(),
                'typical_falsification_type': hypothesis.falsification_type.value,
                'typical_replacement': hypothesis.nuance_added or hypothesis.successor_hypothesis
            }).eq('id', existing.data[0]['id']).execute()
        else:
            # Create new pattern
            store.client.table('collective_falsification_patterns').insert({
                'archetype_name': archetype.name,
                'hypothesis_pattern': hypothesis.hypothesis_type,
                'frequency': 1,
                'common_falsification_type': hypothesis.falsification_type.value,
                'typical_replacement': hypothesis.nuance_added or hypothesis.successor_hypothesis,
                'created_at': datetime.utcnow().isoformat()
            }).execute()


# Integration helper for BehavioralScientistAgent
class HypothesisFalsificationTracker:
    """
    Tracks active hypotheses and detects falsification.
    Used by BehavioralScientistAgent to maintain scientific rigor.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.archive = DisprovenHypothesisArchive()
        self.active_hypotheses: Dict[str, Dict] = {}
    
    async def register_hypothesis(
        self,
        hypothesis_text: str,
        hypothesis_type: str,
        evidence_level: EvidenceLevel,
        expected_observations: List[str]
    ) -> str:
        """
        Register a new hypothesis for tracking.
        Returns hypothesis_id.
        """
        hypothesis_id = f"active_{self.user_id}_{datetime.utcnow().timestamp()}"
        
        self.active_hypotheses[hypothesis_id] = {
            'hypothesis_id': hypothesis_id,
            'hypothesis_text': hypothesis_text,
            'hypothesis_type': hypothesis_type,
            'evidence_level': evidence_level,
            'formed_at': datetime.utcnow(),
            'expected_observations': expected_observations,
            'observations_seen': [],
            'confidence_trajectory': [],
            'status': 'active'
        }
        
        # Check against archive first
        similar_falsified = await self.archive.check_similar_falsifications(
            self.user_id, hypothesis_text
        )
        
        if similar_falsified and similar_falsified[0]['combined_relevance'] > 0.8:
            # High similarity to falsified hypothesis
            return {
                'hypothesis_id': hypothesis_id,
                'warning': 'similar_to_falsified',
                'similar_falsifications': similar_falsified[:3],
                'recommendation': 'formulate_nuanced_variant'
            }
        
        return {'hypothesis_id': hypothesis_id, 'status': 'registered'}
    
    async def record_observation(
        self,
        hypothesis_id: str,
        observation: Dict,
        supports: bool,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Record observation against hypothesis. Check for falsification.
        """
        if hypothesis_id not in self.active_hypotheses:
            return {'error': 'hypothesis_not_found'}
        
        hyp = self.active_hypotheses[hypothesis_id]
        hyp['observations_seen'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'supports': supports,
            'confidence': confidence
        })
        hyp['confidence_trajectory'].append(confidence)
        
        # Check for falsification
        falsification = await self._check_falsification(hypothesis_id)
        
        if falsification:
            # Archive it
            await self.archive.archive_hypothesis(
                user_id=self.user_id,
                original_hypothesis=hyp['hypothesis_text'],
                hypothesis_type=hyp['hypothesis_type'],
                evidence_level=hyp['evidence_level'],
                falsification_type=falsification['type'],
                falsification_evidence=falsification['evidence'],
                successor_hypothesis=falsification.get('successor'),
                nuance_added=falsification.get('nuance')
            )
            
            hyp['status'] = 'falsified'
            
            return {
                'status': 'falsified',
                'falsification_type': falsification['type'],
                'archived': True,
                'lesson': falsification.get('lesson')
            }
        
        # Check for confirmation (promote to belief)
        if len(hyp['observations_seen']) >= 5:
            support_rate = sum(1 for o in hyp['observations_seen'] if o['supports']) / len(hyp['observations_seen'])
            
            if support_rate > 0.8 and confidence > 0.7:
                hyp['status'] = 'confirmed'
                return {
                    'status': 'confirmed',
                    'support_rate': support_rate,
                    'can_enter_belief_network': hyp['evidence_level'].value >= EvidenceLevel.COUNTERFACTUAL_STABILITY.value
                }
        
        return {
            'status': 'active',
            'observations_count': len(hyp['observations_seen']),
            'current_confidence': confidence
        }
    
    async def _check_falsification(self, hypothesis_id: str) -> Optional[Dict]:
        """Check if hypothesis should be falsified."""
        hyp = self.active_hypotheses[hypothesis_id]
        observations = hyp['observations_seen']
        
        if len(observations) < 3:
            return None
        
        # Check for contradictory evidence
        recent = observations[-5:]
        support_rate = sum(1 for o in recent if o['supports']) / len(recent)
        
        if support_rate < 0.2 and len(recent) >= 3:
            # Strong contradictory evidence
            return {
                'type': FalsificationType.CONTRADICTORY_EVIDENCE,
                'evidence': {
                    'recent_observations': recent,
                    'support_rate': support_rate,
                    'hypothesis_formed_at': hyp['formed_at'].isoformat(),
                    'supporting_count': sum(1 for o in observations if o['supports'])
                },
                'lesson': 'Pattern did not hold under continued observation'
            }
        
        # Check for confidence collapse
        if len(hyp['confidence_trajectory']) >= 3:
            recent_conf = hyp['confidence_trajectory'][-3:]
            if all(c < 0.3 for c in recent_conf):
                return {
                    'type': FalsificationType.BASELINE_REGRESSION,
                    'evidence': {
                        'confidence_trajectory': hyp['confidence_trajectory'],
                        'final_confidence': recent_conf[-1]
                    },
                    'lesson': 'Initial pattern was likely noise or regression to mean'
                }
        
        return None
