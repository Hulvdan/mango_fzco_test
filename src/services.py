from contextlib import asynccontextmanager

from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import func, select

from .db import Group, GroupParticipant, User

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DomainError(Exception):
    status: int


class UserDoesNotExistError(DomainError):
    status = 404


class WrongPasswordError(DomainError):
    status = 404


class Login(BaseModel):
    email: str
    password: str


async def login(data: Login, session) -> int:
    user = (await session.execute(select(User).where(User.email == data.email))).scalar()
    if user is None:
        raise UserDoesNotExistError

    if not _pwd_context.verify(data.password, user.password):
        raise WrongPasswordError

    return user.id


async def make_user(*, email: str, name: str, password: str, session) -> int:
    user = User(email=email, name=name, password=_pwd_context.hash(password))
    session.add(user)
    await session.commit()
    return user.id


class MakeGroup(BaseModel):
    name: str
    participant_user_ids: list[int]


class MakeGroupTooManyParticipantsError(DomainError):
    status = 400


class MakeGroupSomeParticipantsDontExistError(DomainError):
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
        raise MakeGroupTooManyParticipantsError

    existing_users_count = (
        await session.execute(
            select(func.count(User.id))
            .select_from(User)
            .where(User.id.in_(data.participant_user_ids))
        )
    ).scalar()

    if existing_users_count != len(data.participant_user_ids):
        raise MakeGroupSomeParticipantsDontExistError

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
