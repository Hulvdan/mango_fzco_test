import bcrypt
from pydantic import BaseModel

from .db import Group, GroupParticipant, User
from .settings import settings


async def make_user(*, email: str, name: str, password: str, session):
    salt = settings.service_password_salt.encode("utf-8")
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    user = User(email=email, name=name, password=hashed_password.decode())
    session.add(user)
    await session.commit()


class MakeGroup(BaseModel):
    name: str
    participant_user_ids: list[int]


MAKE_GROUP_ERROR_TOO_MUCH_PARTICIPANTS = 1
MAKE_GROUP_ERROR_SOME_PARTICIPANTS_DONT_EXIST = 2


# NOTE: В make_group 2 стадии:
#
# 1) Проверка существования перечисленных в participant_user_ids пользователей.
# 2) Создание группы с привязкой участников.
#
# Можно разделить session на read_only_session и write_session.
# Для валидации - ходили бы в реплику. Так мы снизили бы нагрузку на master БД.
async def make_group(data: MakeGroup, creator_id: int, session):
    # TODO validate all participants exist
    if len(data.participant_user_ids) > 100:
        return

    group = Group(
        name=data.name,
        creator_id=creator_id,
    )
    session.add(group)
    await session.flush()

    session.add_all(
        GroupParticipant(group_id=group.id, user_id=user_id)
        for user_id in data.participant_user_ids
    )
    await session.commit()
