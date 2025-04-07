from contextlib import asynccontextmanager
from datetime import datetime

from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import exists, func, select

from .db import Chat, ChatParticipant, Group, Message, User

_password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DomainError(Exception):
    status: int


class NotAuthorizedError(DomainError):
    status = 403


class UserDoesNotExistError(DomainError):
    status = 404


class WrongPasswordError(DomainError):
    status = 404


class LoginData(BaseModel):
    email: str
    password: str


async def login(data: LoginData, session) -> int:
    user = (await session.execute(select(User).where(User.email == data.email))).scalar()
    if user is None:
        raise UserDoesNotExistError

    if not _password_context.verify(data.password, user.password):
        raise WrongPasswordError

    return user.id


async def make_user(*, email: str, name: str, password: str, session) -> int:
    user = User(email=email, name=name, password=_password_context.hash(password))
    session.add(user)
    await session.commit()
    return user.id


class MakeGroupData(BaseModel):
    name: str
    participant_user_ids: list[int]


class MakeGroupTooManyParticipantsError(DomainError):
    status = 400


class MakeGroupSomeParticipantsDontExistError(DomainError):
    status = 404


class MakeGroupResponse(BaseModel):
    group_id: int
    chat_id: int


# В make_group 2 стадии:
#
# 1) Проверка существования перечисленных в participant_user_ids пользователей.
# 2) Создание группы с привязкой участников.
#
# Можно разделить session на read_only_session и write_session.
# Для валидации - ходили бы в реплику. Так мы снизили бы нагрузку на master БД.
async def make_group(data: MakeGroupData, creator_id: int, session) -> MakeGroupResponse:
    if creator_id not in data.participant_user_ids:
        data.participant_user_ids.append(creator_id)

    # Для меня странным кажется то, что человек бы разом
    # больше 100 участников бы добавлял.
    # Я бы потом добавил функцию добавления ещё участников.
    if len(data.participant_user_ids) > 100:
        raise MakeGroupTooManyParticipantsError

    existing_users_count = (
        await session.execute(
            select(func.count(User.id))
            .select_from(User)
            .where(User.id.in_(data.participant_user_ids))
        )
    ).scalar()

    # Считаем, что клиент не прав, если пытается
    # добавлять несуществующих участников.
    if existing_users_count != len(data.participant_user_ids):
        raise MakeGroupSomeParticipantsDontExistError

    chat = Chat(type=Chat.TYPE_GROUP)
    session.add(chat)
    await session.flush()

    group = Group(name=data.name, creator_id=creator_id, chat_id=chat.id)
    session.add(group)
    await session.flush()

    session.add_all(
        ChatParticipant(chat_id=chat.id, user_id=user_id)
        for user_id in data.participant_user_ids
    )
    await session.commit()

    return MakeGroupResponse(
        group_id=group.id,
        chat_id=group.chat_id,
    )


class MakeMessageData(BaseModel):
    text: str

    # # Для предотвращения дублирования при параллельной отправке.
    # # Разом на 3 сервиса можно было бы отправить запрос
    # # с одним и тем же timestamp от клиента.
    # timestamp: int


class MessageData(BaseModel):
    id: int
    chat_id: int
    sender_id: int
    text: str
    timestamp: datetime
    is_read: bool


# async def message_user():
#     # см. Chat.user_ids
#     user1 = sender_id
#     user2 = data.entity_id
#     if user2 < user1:
#         user1, user2 = user2, user1
#     user_ids = user1 << 32 + user2
#
#     st = (
#         insert(Chat).values(user_ids=user_ids).on_conflict_do_nothing().returning(Chat.id)
#     )
#     chat_id = (await session.execute(st)).scalar()


class GroupDoesNotExistError(DomainError):
    status = 404


async def message_group(
    *, data: MakeMessageData, sender_id: int, group_id: int, session
) -> tuple[MessageData, list[int]]:
    chat_id = (
        await session.execute(select(Group.chat_id).filter(Group.id == group_id))
    ).scalar()

    if chat_id is None:
        raise GroupDoesNotExistError

    message = Message(
        chat_id=chat_id,
        sender_id=sender_id,
        text=data.text,
    )
    session.add(message)
    await session.commit()

    return MessageData(
        id=message.id,
        chat_id=message.chat_id,
        sender_id=message.sender_id,
        text=message.text,
        timestamp=message.timestamp,
        is_read=False,
    )


class UserCantAccessSpecifiedChatError(DomainError):
    status = 403


async def history(
    *, chat_id: int, user_id: int, later_that_message_id: int, limit: int, session
) -> list[MessageData]:
    can_access_chat = (
        await session.execute(
            select(ChatParticipant)
            .where(ChatParticipant.chat_id == chat_id)
            .where(ChatParticipant.user_id == user_id)
            .limit(1)
        )
    ).scalar()
    if not can_access_chat:
        raise UserCantAccessSpecifiedChatError

    statement = select(Message).where(Message.chat_id == chat_id)
    if later_that_message_id:
        statement = statement.where(Message.id > later_that_message_id)

    # «Сообщения должны быть отсортированы по времени отправки (по возрастанию)»
    # Отсортировал по возрастанию id - не нужно вешать отдельный индекс на timestamp.
    # Эффект в данном случае тот же.
    #
    # Не совсем понимаю, почему «по возрастанию». В телеграмме я по-идее бы делал так,
    # чтобы выводились сперва самые новые сообщения. А историю мы бы крутили вспять.
    messages = (
        await session.execute(statement.order_by(Message.id.asc()).limit(limit))
    ).scalars()

    return [
        MessageData(
            id=i.id,
            chat_id=i.chat_id,
            sender_id=i.sender_id,
            text=i.text,
            timestamp=i.timestamp,
            # TODO
            is_read=False,
        )
        for i in messages
    ]


async def can_access_to_chat(*, chat_id: int, user_id: int, session) -> bool:
    return (
        await session.execute(
            select(exists().select_from(ChatParticipant)).where(
                ChatParticipant.chat_id == chat_id, ChatParticipant.user_id == user_id
            )
        )
    ).scalar()


async def fill_db_with_initial_data():
    from .db import make_session, reinit_db_from_scratch

    await reinit_db_from_scratch()

    async with asynccontextmanager(make_session)() as session:
        for i in range(1, 4):
            await make_user(
                email=f"test{i}@test.com",
                name=f"test{i}",
                password="test",
                session=session,
            )
