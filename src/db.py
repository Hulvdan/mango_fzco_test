
import sqlalchemy as sa
from sqlalchemy import CHAR, Column, ForeignKey, Integer, String
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
    type = Column(sa.SmallInteger)


class Group(_Table):
    __tablename__ = "groups"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = String(50)
    creator_id = Column(Integer, ForeignKey("users.id"))
    # id, название, создатель, список участников.

    # participants =
    # children = relationship("Child", back_populates="parent")


class GroupParticipant(_Table):
    __tablename__ = "group_participants"

    group_id = Column(Integer, ForeignKey("groups.id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)


class Message(_Table):
    __tablename__ = "messages"

    id = Column(Integer, autoincrement=True, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    # id, chat_id, sender_id, text, timestamp, прочитано


_engine = create_async_engine(settings.postgres_dsn)

_sessionmaker = sa.orm.sessionmaker(
    bind=_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=AsyncSession,
)


# По-идее эта штука вполне могла бы await-ить,
# пока не получим какой-то освободившийся коннект в пуле.
async def make_session():
    session = _sessionmaker()
    try:
        yield session
    finally:
        await session.close()


async def reinit_db_from_scratch():
    async with _engine.begin() as conn:
        await conn.run_sync(_Table.metadata.drop_all)
        await conn.run_sync(_Table.metadata.create_all)
