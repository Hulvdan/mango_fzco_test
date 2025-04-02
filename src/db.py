import sqlalchemy as sa
from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.declarative import declarative_base


class DbSettings(BaseSettings):
    db: str = Field(str)
    host: str = Field(str)
    user: str = Field(str)
    password: str = Field(str)

    def dns(self) -> str:
        return "postgresql+asyncpg://{}:{}@{}:5432/{}".format(
            self.user, self.db, self.host, self.password
        )


Table = declarative_base()


class User(Table):
    __tablename__ = "users"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String(50))

    password = Column(String())


class Chat(Table):
    __tablename__ = "chats"

    TYPE_PERSONAL = 0
    TYPE_GROUP = 1

    id = Column(Integer, autoincrement=True, primary_key=True)
    type = Column(sa.SmallInteger)


class Group(Table):
    __tablename__ = "groups"

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = String(50)
    creator_id = Column(Integer, ForeignKey("users.id"))
    # id, название, создатель, список участников.

    # participants =
    # children = relationship("Child", back_populates="parent")


class GroupParticipant(Table):
    __tablename__ = "group_participants"

    group_id = Column(Integer, ForeignKey("groups.id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)


class Message(Table):
    __tablename__ = "messages"

    id = Column(Integer, autoincrement=True, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    # id, chat_id, sender_id, text, timestamp, прочитано


engine = create_async_engine(DbSettings(_env_prefix="POSTGRES_", _env_file=".env").dns())
session = sa.orm.sessionmaker(bind=engine)()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Table.metadata.drop_all)
