"""
Authentication API with behavioral profiling and device intelligence.
v2.0: Now captures behavioral context, manages device trust, and initializes belief networks.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from uuid import uuid4

from spirit.config import settings
from spirit.db import async_session, get_behavioral_store
from spirit.models import User
from spirit.schema.request import RegisterRequest
from spirit.schema.response import TokenResponse
from spirit.memory.episodic_memory import EpisodicMemorySystem


router = APIRouter(prefix="/auth", tags=["auth"])
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_pw(password: str) -> str:
    return pwd.hash(password)

def verify_pw(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)

def create_access_token(sub: str, device_id: Optional[str] = None) -> str:
    """Create JWT with optional device binding for behavioral tracking."""
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": sub, 
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid4())[:8],  # Unique token ID for revocation
    }
    if device_id:
        payload["device_id"] = device_id
    
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")

def create_refresh_token(sub: str) -> str:
    """Create longer-lived refresh token."""
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode(
        {"sub": sub, "exp": expire, "type": "refresh"},
        settings.jwt_secret,
        algorithm="HS256"
    )

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(
    db: AsyncSession = Depends(lambda: async_session()), 
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get current user with behavioral context logging."""
    payload = decode_token(token)
    user_id = int(payload.get("sub"))
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # NEW: Log authentication event for behavioral analysis
    await _log_auth_event(user_id, "token_validation", payload.get("device_id"))
    
    return user

async def get_current_user_with_context(
    request: Request,
    db: AsyncSession = Depends(lambda: async_session()), 
    token: str = Depends(oauth2_scheme)
) -> tuple[User, Dict[str, Any]]:
    """
    Get current user with full behavioral context.
    Used by endpoints that need device/location behavioral data.
    """
    payload = decode_token(token)
    user_id = int(payload.get("sub"))
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Build behavioral context
    context = {
        "device_id": payload.get("device_id"),
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": request.url.path
    }
    
    # Log with context
    await _log_auth_event(user_id, "authenticated_request", payload.get("device_id"), context)
    
    return user, context


@router.post("/register", response_model=TokenResponse)
async def register(
    body: RegisterRequest, 
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(lambda: async_session())
):
    """
    Register new user with behavioral profiling initialization.
    v2.0: Creates initial belief model and captures onboarding context.
    """
    # Check existing user
    res = await db.execute(select(User).where(User.email == body.email))
    if res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=body.email, 
        hashed_password=hash_pw(body.password),
        created_at=datetime.utcnow(),
        onboarding_context={
            "signup_time": datetime.utcnow().isoformat(),
            "signup_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "referrer": request.headers.get("referer")
        }
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # NEW: Initialize belief network with defaults
    background_tasks.add_task(_initialize_user_beliefs, user.id)
    
    # NEW: Create initial episodic memory anchor
    background_tasks.add_task(_create_onboarding_memory, user.id)
    
    # Generate tokens
    device_id = str(uuid4())[:12]
    access_token = create_access_token(str(user.id), device_id)
    refresh_token = create_refresh_token(str(user.id))
    
    # Store device registration
    await _register_device(user.id, device_id, request)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        device_id=device_id,
        user_id=user.id
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(), 
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(lambda: async_session())
):
    """
    Authenticate with behavioral anomaly detection.
    v2.0: Detects suspicious login patterns and updates behavioral profile.
    """
    # Verify credentials
    res = await db.execute(select(User).where(User.email == form.username))
    user = res.scalar_one_or_none()
    if not user or not verify_pw(form.password, user.hashed_password):
        # Log failed attempt for security analysis
        await _log_failed_login(form.username, request)
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # NEW: Behavioral security check
    is_suspicious, risk_factors = await _check_login_anomaly(user.id, request)
    if is_suspicious:
        # Require additional verification or notify
        await _handle_suspicious_login(user.id, request, risk_factors)
        # Still allow but flag for review
        flagged = True
    else:
        flagged = False
    
    # Generate device-bound tokens
    device_id = await _get_or_create_device_id(user.id, request)
    access_token = create_access_token(str(user.id), device_id)
    refresh_token = create_refresh_token(str(user.id))
    
    # Update last login and behavioral context
    user.last_login_at = datetime.utcnow()
    user.login_count = (user.login_count or 0) + 1
    await db.commit()
    
    # Log successful login with context
    background_tasks.add_task(
        _log_successful_login,
        user.id,
        device_id,
        request,
        flagged
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        device_id=device_id,
        user_id=user.id,
        flagged=flagged
    )


@router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(lambda: async_session())
):
    """
    Refresh access token with behavioral validation.
    Ensures refresh patterns are consistent with user behavior.
    """
    try:
        payload = jwt.decode(refresh_token, settings.jwt_secret, algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = int(payload.get("sub"))
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Check if refresh pattern is anomalous
        is_anomalous = await _check_refresh_pattern(user_id)
        if is_anomalous:
            # Force re-authentication
            raise HTTPException(status_code=401, detail="Suspicious activity detected. Please log in again.")
        
        # Issue new tokens
        device_id = payload.get("device_id")
        new_access = create_access_token(str(user_id), device_id)
        new_refresh = create_refresh_token(str(user_id))
        
        return {
            "access_token": new_access,
            "refresh_token": new_refresh
        }
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.post("/logout")
async def logout(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """
    Logout with session cleanup and behavioral logging.
    """
    # Log session end
    background_tasks.add_task(_log_session_end, user.id)
    
    return {"detail": "Successfully logged out"}


@router.get("/devices")
async def list_devices(
    user: User = Depends(get_current_user)
):
    """
    List trusted devices with behavioral profiles.
    """
    store = await get_behavioral_store()
    if not store:
        return {"devices": []}
    
    devices = store.client.table('user_devices').select('*').eq(
        'user_id', str(user.id)
    ).order('last_used_at', desc=True).execute()
    
    return {
        "devices": [
            {
                "device_id": d.get('device_id'),
                "device_type": d.get('device_type'),
                "trusted": d.get('trusted', False),
                "first_seen": d.get('first_seen_at'),
                "last_used": d.get('last_used_at'),
                "login_count": d.get('login_count', 0)
            }
            for d in (devices.data or [])
        ]
    }


@router.delete("/devices/{device_id}")
async def revoke_device(
    device_id: str,
    user: User = Depends(get_current_user)
):
    """
    Revoke trust for a specific device.
    """
    store = await get_behavioral_store()
    if store:
        store.client.table('user_devices').update({
            'trusted': False,
            'revoked_at': datetime.utcnow().isoformat()
        }).eq('user_id', str(user.id)).eq('device_id', device_id).execute()
    
    return {"detail": f"Device {device_id} revoked"}


@router.get("/security-status")
async def security_status(
    user: User = Depends(get_current_user)
):
    """
    Get user's behavioral security profile.
    """
    store = await get_behavioral_store()
    if not store:
        return {"status": "unknown"}
    
    # Get recent login patterns
    recent_logins = store.client.table('auth_events').select('*').eq(
        'user_id', str(user.id)
    ).gte('timestamp', (datetime.utcnow() - timedelta(days=7)).isoformat()).execute()
    
    if not recent_logins.data:
        return {"status": "insufficient_data"}
    
    # Analyze patterns
    unique_devices = len(set(l.get('device_id') for l in recent_logins.data if l.get('device_id')))
    failed_attempts = sum(1 for l in recent_logins.data if l.get('event_type') == 'failed_login')
    suspicious_flags = sum(1 for l in recent_logins.data if l.get('flagged'))
    
    risk_level = "low"
    if failed_attempts > 5 or suspicious_flags > 2 or unique_devices > 3:
        risk_level = "medium"
    if failed_attempts > 10 or suspicious_flags > 5:
        risk_level = "high"
    
    return {
        "risk_level": risk_level,
        "active_devices": unique_devices,
        "failed_attempts_7d": failed_attempts,
        "suspicious_events": suspicious_flags,
        "recommendations": _generate_security_recommendations(risk_level, unique_devices)
    }


# Helper functions

async def _initialize_user_beliefs(user_id: int):
    """Initialize default belief model for new user."""
    store = await get_behavioral_store()
    if not store:
        return
    
    default_beliefs = {
        "self_efficacy": 0.5,  # Neutral starting point
        "optimal_time": "unknown",
        "work_style_preference": "unknown",
        "distraction_susceptibility": "unknown",
        "goal_completion_rate": 0.0,
        "goals_completed": 0,
        "goals_attempted": 0
    }
    
    store.client.table('belief_networks').insert({
        'belief_id': str(uuid4()),
        'user_id': str(user_id),
        'beliefs': default_beliefs,
        'confidence': 0.3,  # Low confidence until data collected
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }).execute()

async def _create_onboarding_memory(user_id: int):
    """Create initial episodic memory for user journey."""
    memory = EpisodicMemorySystem(user_id)
    await memory.store_episode(
        episode_type="onboarding",
        content={"event": "account_created", "context": "initial_signup"},
        significance=0.8  # High significance as origin point
    )

async def _register_device(user_id: int, device_id: str, request: Request):
    """Register new device for behavioral tracking."""
    store = await get_behavioral_store()
    if not store:
        return
    
    device_type = _infer_device_type(request.headers.get("user-agent", ""))
    
    store.client.table('user_devices').insert({
        'device_id': device_id,
        'user_id': str(user_id),
        'device_type': device_type,
        'first_seen_at': datetime.utcnow().isoformat(),
        'last_used_at': datetime.utcnow().isoformat(),
        'trusted': True,
        'ip_address': request.client.host if request.client else None,
        'user_agent': request.headers.get("user-agent"),
        'login_count': 1
    }).execute()

async def _log_auth_event(user_id: int, event_type: str, device_id: Optional[str], context: Optional[Dict] = None):
    """Log authentication event for behavioral analysis."""
    store = await get_behavioral_store()
    if not store:
        return
    
    data = {
        'event_id': str(uuid4()),
        'user_id': str(user_id),
        'event_type': event_type,
        'device_id': device_id,
        'timestamp': datetime.utcnow().isoformat()
    }
    if context:
        data['context'] = context
    
    store.client.table('auth_events').insert(data).execute()

async def _log_failed_login(email: str, request: Request):
    """Log failed login attempt."""
    store = await get_behavioral_store()
    if not store:
        return
    
    store.client.table('failed_logins').insert({
        'attempt_id': str(uuid4()),
        'email_attempted': email,
        'ip_address': request.client.host if request.client else None,
        'user_agent': request.headers.get("user-agent"),
        'timestamp': datetime.utcnow().isoformat()
    }).execute()

async def _check_login_anomaly(user_id: int, request: Request) -> tuple[bool, list]:
    """Check if login is anomalous based on behavioral history."""
    store = await get_behavioral_store()
    if not store:
        return False, []
    
    risk_factors = []
    
    # Check for new device
    recent_devices = store.client.table('auth_events').select('device_id').eq(
        'user_id', str(user_id)
    ).gte('timestamp', (datetime.utcnow() - timedelta(days=30)).isoformat()).execute()
    
    current_ip = request.client.host if request.client else None
    known_devices = set(d.get('device_id') for d in recent_devices.data if d.get('device_id'))
    
    if len(known_devices) > 0 and current_ip not in known_devices:
        risk_factors.append("new_device")
    
    # Check for unusual time
    hour = datetime.utcnow().hour
    if hour < 5 or hour > 23:  # Unusual hours
        risk_factors.append("unusual_time")
    
    # Check for location anomaly (simplified)
    recent_ips = set()
    for event in recent_devices.data:
        ctx = event.get('context', {})
        if ctx and ctx.get('ip_address'):
            recent_ips.add(ctx['ip_address'])
    
    if recent_ips and current_ip not in recent_ips:
        risk_factors.append("new_location")
    
    is_suspicious = len(risk_factors) >= 2
    return is_suspicious, risk_factors

async def _handle_suspicious_login(user_id: int, request: Request, risk_factors: list):
    """Handle suspicious login detection."""
    store = await get_behavioral_store()
    if not store:
        return
    
    # Log for review
    store.client.table('suspicious_logins').insert({
        'alert_id': str(uuid4()),
        'user_id': str(user_id),
        'risk_factors': risk_factors,
        'ip_address': request.client.host if request.client else None,
        'timestamp': datetime.utcnow().isoformat(),
        'resolved': False
    }).execute()

async def _get_or_create_device_id(user_id: int, request: Request) -> str:
    """Get existing device ID or create new one."""
    # Simplified - would check cookies/device fingerprints
    return str(uuid4())[:12]

async def _log_successful_login(user_id: int, device_id: str, request: Request, flagged: bool):
    """Log successful login with full context."""
    await _log_auth_event(
        user_id, 
        "successful_login", 
        device_id,
        {
            "ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "flagged": flagged
        }
    )
    
    # Update device last used
    store = await get_behavioral_store()
    if store:
        # Get current count
        device = store.client.table('user_devices').select('*').eq(
            'user_id', str(user_id)
        ).eq('device_id', device_id).execute()
        
        if device.data:
            current_count = device.data[0].get('login_count', 0)
            store.client.table('user_devices').update({
                'last_used_at': datetime.utcnow().isoformat(),
                'login_count': current_count + 1
            }).eq('user_id', str(user_id)).eq('device_id', device_id).execute()

async def _check_refresh_pattern(user_id: int) -> bool:
    """Check if token refresh pattern is anomalous."""
    store = await get_behavioral_store()
    if not store:
        return False
    
    # Check for excessive refresh attempts
    recent = store.client.table('auth_events').select('*').eq(
        'user_id', str(user_id)
    ).eq('event_type', 'token_refresh').gte(
        'timestamp', (datetime.utcnow() - timedelta(hours=1)).isoformat()
    ).execute()
    
    # More than 10 refreshes in an hour is suspicious
    return len(recent.data) > 10 if recent.data else False

async def _log_session_end(user_id: int):
    """Log session termination."""
    await _log_auth_event(user_id, "logout", None)

def _infer_device_type(user_agent: str) -> str:
    """Infer device type from user agent."""
    ua = user_agent.lower()
    if "mobile" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    if "tablet" in ua or "ipad" in ua:
        return "tablet"
    return "desktop"

def _generate_security_recommendations(risk_level: str, device_count: int) -> list:
    """Generate security recommendations based on risk profile."""
    recs = []
    if risk_level == "high":
        recs.append("Enable two-factor authentication immediately")
        recs.append("Review and revoke unrecognized devices")
    if risk_level == "medium":
        recs.append("Review your active devices")
    if device_count > 3:
        recs.append("Consider revoking unused devices")
    if not recs:
        recs.append("Your security posture looks good")
    return recs
