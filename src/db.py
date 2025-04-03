from contextlib import asynccontextmanager
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import (
    CHAR,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from .settings import settings

_Table = declarative_base()


class User(_Table):
    __tablename__ = "users"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(50))
    email = Column(String(50), unique=True)
    password = Column(CHAR(60))  # Храним hash размером 60 байт


class Chat(_Table):
    __tablename__ = "chats"

    TYPE_PERSONAL = 0
    TYPE_GROUP = 1

    id = Column(Integer, autoincrement=True, primary_key=True)
    type = Column(SmallInteger)

    # Если это чат пользователей,
    # здесь будет user1_id << 32 + user2_id (user1_id, user2_id отсортированы).
    user_ids = Column(BigInteger, unique=True, nullable=True)

    __table_args__ = (
        Index(
            "ix_user_ids",
            "user_ids",
            user_ids != None,
        ),
    )

    # Честно говоря, очень хотел упаковать всё в одну колонку int64 id.
    #
    # Если это был бы чат двух пользователей,
    # то в первых 4 байтах бы хранился id первого пользователя,
    # во вторых - второго.
    #
    # Если бы это был групповой чат, первые 4 байта всегда были бы 0.
    # Создание нового группового чата назначало бы id через Sequence.
    #
    # Решил не выпендриваться из-за «Соответствие всем указанным требованиям».


class Group(_Table):
    __tablename__ = "groups"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = String(50)
    creator_id = Column(Integer, ForeignKey("users.id"))
    chat_id = Column(Integer, ForeignKey("chats.id"))


class ChatParticipant(_Table):
    __tablename__ = "chat_participants"

    chat_id = Column(Integer, ForeignKey("chats.id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)


class Message(_Table):
    __tablename__ = "messages"

    id = Column(Integer, autoincrement=True, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    sender_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # id, chat_id, sender_id, text, timestamp, прочитано

    __table_args__ = (Index("ix_chat_id", "chat_id"),)


_engine = None
_sessionmaker = None


# Для pytest `custom_poolclass` ставится NullPool, чтобы не было проблем с asyncio.
def init_engine_and_sessionmaker(*, custom_poolclass=None):
    global _engine
    global _sessionmaker

    assert _engine is None
    assert _sessionmaker is None

    _engine = create_async_engine(settings.postgres_dsn, poolclass=custom_poolclass)
    _sessionmaker = sa.orm.sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def make_session():
    assert _sessionmaker is not None

    session = _sessionmaker()
    try:
        yield session
    finally:
        await session.close()


async def reinit_db_from_scratch():
    assert _engine is not None

    async with asynccontextmanager(make_session)() as session:
        await session.execute(sa.text("DROP SCHEMA IF EXISTS public CASCADE"))
        await session.execute(sa.text("CREATE SCHEMA public"))

    async with _engine.begin() as conn:
        await conn.run_sync(_Table.metadata.create_all)
