"""
Privacy validation and filtering for behavioral data.
Defense-in-depth: validates edge device compliance with privacy rules.
"""

from datetime import datetime, timedelta

from spirit.models.behavioral import (
    ScreenTimeSession, 
    BehavioralObservation, 
    PrivacyLevel,
    AppCategory
)


class PrivacyFilter:
    """
    Validates that incoming data respects privacy constraints.
    Runs server-side as defense in depth against misconfigured edge devices.
    """
    
    # Apps that should never have their package names logged
    SENSITIVE_APP_PATTERNS = [
        'dating', 'therapy', 'medical', 'banking', 'password',
        'vpn', 'tor', 'signal', 'whatsapp', 'messenger'
    ]
    
    # Minimum aggregation window (seconds)
    MIN_AGGREGATION_WINDOW = 55
    
    def validate(self, session: ScreenTimeSession) -> bool:
        """
        Validate screen session meets privacy requirements.
        Returns True if acceptable, False if must be rejected.
        """
        # 1. Check for raw PII in app names
        if session.app_package:
            lower_package = session.app_package.lower()
            for pattern in self.SENSITIVE_APP_PATTERNS:
                if pattern in lower_package:
                    # Should have been hashed by edge device
                    return False
        
        # 2. Verify app category is present (required for aggregation)
        if not session.app_category or session.app_category == AppCategory.OTHER:
            # Edge device failed to classify—reject to force reprocessing
            return False
        
        # 3. Check duration is reasonable (filter glitch data)
        if session.duration_seconds is not None:
            if session.duration_seconds < 0:
                return False
            if session.duration_seconds > 86400:  # 24h
                return False
        
        # 4. Verify hashing (app_name_hash should be present)
        if not session.app_name_hash:
            return False
        
        # 5. Check data quality score
        if session.data_quality_score < 0.5:
            return False
        
        return True
    
    def validate_observation(self, observation: BehavioralObservation) -> bool:
        """
        Validate behavioral observation meets privacy standards.
        """
        # 1. Verify privacy level is set
        if not observation.privacy_level:
            return False
        
        # 2. Restricted/Sensitive data should not reach server
        if observation.privacy_level in [PrivacyLevel.RESTRICTED, PrivacyLevel.SENSITIVE]:
            return False
        
        # 3. Check for location precision in context
        if 'gps_lat' in observation.context or 'gps_lon' in observation.context:
            return False
        
        if 'location_accuracy_meters' in observation.context:
            accuracy = observation.context['location_accuracy_meters']
            if accuracy < 1000:  # Less than 1km is too precise
                return False
        
        # 4. Verify no raw communication content
        behavior = observation.behavior
        if 'message_content' in behavior or 'notification_text' in behavior:
            return False
        
        # 5. Check timestamp is reasonable
        now = datetime.utcnow()
        if observation.timestamp > now + timedelta(minutes=5):
            return False  # Future timestamp
        if observation.timestamp < now - timedelta(days=7):
            return False  # Too old
        
        return True
    
    def calculate_privacy_budget_consumption(
        self,
        observation: BehavioralObservation
    ) -> float:
        """
        Calculate epsilon consumption for differential privacy.
        More granular data = higher privacy cost.
        """
        epsilon = 0.01  # Base cost
        
        behavior = observation.behavior
        
        # Granularity penalties
        if 'exact_duration_sec' in behavior:
            epsilon += 0.05
        
        if 'scroll_velocity' in behavior:
            epsilon += 0.02
        
        if 'typing_pattern' in behavior:
            epsilon += 0.1
        
        # Time precision
        ts = observation.timestamp
        if ts.second != 0 or ts.microsecond != 0:
            epsilon += 0.03
        
        return epsilon
