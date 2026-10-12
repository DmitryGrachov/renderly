import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from .config import JWT_SECRET, JWT_TTL_DAYS

_ph = PasswordHasher()
# для неизвестного email всё равно считаем хеш — чтобы время ответа не выдавало, есть ли аккаунт
DUMMY_HASH = _ph.hash("inego-timing-equalizer")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _ph.check_needs_rehash(password_hash)


def create_token(user_id: uuid.UUID) -> str:
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(days=JWT_TTL_DAYS)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(jwt.decode(token, JWT_SECRET, algorithms=["HS256"])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
