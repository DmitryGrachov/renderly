import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .config import COOKIE_SECURE, JWT_TTL_DAYS
from .db import get_session
from .models import User
from .security import DUMMY_HASH, create_token, decode_token, hash_password, needs_rehash, verify_password

COOKIE_NAME = "inego_session"

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name is empty")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    name: str
    created_at: datetime


def _set_session(response: Response, user: User) -> None:
    response.set_cookie(
        COOKIE_NAME,
        create_token(user.id),
        max_age=JWT_TTL_DAYS * 24 * 3600,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> User:
    token = request.cookies.get(COOKIE_NAME)
    user_id = decode_token(token) if token else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not_authenticated")
    return user


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: RegisterIn, response: Response, session: AsyncSession = Depends(get_session)):
    user = User(email=data.email.lower(), name=data.name, password_hash=hash_password(data.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "email_taken")
    _set_session(response, user)
    return user


@router.post("/login", response_model=UserOut)
async def login(data: LoginIn, response: Response, session: AsyncSession = Depends(get_session)):
    user = await session.scalar(select(User).where(User.email == data.email.lower()))
    if user is None:
        verify_password(DUMMY_HASH, data.password)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid_credentials")
    if not verify_password(user.password_hash, data.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid_credentials")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(data.password)
        await session.commit()
    _set_session(response, user)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout():
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
    return response


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)):
    return user
