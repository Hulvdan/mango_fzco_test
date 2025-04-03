from contextlib import asynccontextmanager, contextmanager

from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select

from .db import Group, GroupParticipant, User

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DomainException(Exception):
    status: int


class UserDoesNotExist(DomainException):
    status = 404


class WrongPassword(DomainException):
    status = 404


class Login(BaseModel):
    email: str
    password: str


async def login(data: Login, session) -> int:
    user = (await session.execute(select(User).where(User.email == data.email))).scalar()
    if user is None:
        raise UserDoesNotExist

    if not _pwd_context.verify(data.password, user.password):
        raise WrongPassword

    return user.id


async def make_user(*, email: str, name: str, password: str, session) -> int:
    user = User(email=email, name=name, password=_pwd_context.hash(password))
    session.add(user)
    await session.commit()
    return user.id


class MakeGroup(BaseModel):
    name: str
    participant_user_ids: list[int]


class MakeGroupTooManyParticipantsProvided(DomainException):
    status = 400


class MakeGroupSomeParticipantsDontExist(DomainException):
    status = 404


# В make_group 2 стадии:
#
# 1) Проверка существования перечисленных в participant_user_ids пользователей.
# 2) Создание группы с привязкой участников.
#
# Можно разделить session на read_only_session и write_session.
# Для валидации - ходили бы в реплику. Так мы снизили бы нагрузку на master БД.
async def make_group(data: MakeGroup, creator_id: int, session) -> int:
    if creator_id not in data.participant_user_ids:
        data.participant_user_ids.append(creator_id)

    if len(data.participant_user_ids) > 100:
        raise MakeGroupTooManyParticipantsProvided

    # TODO validate all participants exist

    group = Group(name=data.name, creator_id=creator_id)
    session.add(group)
    await session.flush()

    session.add_all(
        GroupParticipant(group_id=group.id, user_id=user_id)
        for user_id in data.participant_user_ids
    )
    await session.commit()

    return group.id


async def fill_db_with_initial_data():
    from .db import make_session, reinit_db_from_scratch

    await reinit_db_from_scratch()

    async with asynccontextmanager(make_session)() as session:
        for i in range(3):
            await make_user(
                email=f"test{i}@test.com",
                name=f"test{i}",
                password="test",
                session=session,
            )
