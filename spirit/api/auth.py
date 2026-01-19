from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from spirit.config import settings
from spirit.db import async_session
from spirit.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_pw(password: str) -> str:
    return pwd.hash(password)

def verify_pw(password: str, hashed: str) -> bool:
    return pwd.verify(password, hashed)

def create_access_token(data: dict) -> str:
    return jwt.encode(data, settings.jwt_secret, algorithm="HS256")

@router.post("/register")
async def register(email: str, password: str, db: AsyncSession = Depends(lambda: async_session())):
    res = await db.execute(select(User).where(User.email == email))
    if res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="email taken")
    user = User(email=email, hashed_password=hash_pw(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"user_id": user.id, "token": create_access_token({"sub": str(user.id)})}

@router.post("/login")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(lambda: async_session())):
    res = await db.execute(select(User).where(User.email == form.username))
    user = res.scalar_one_or_none()
    if not user or not verify_pw(form.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="invalid credentials")
    return {"access_token": create_access_token({"sub": str(user.id)}), "token_type": "bearer"}
