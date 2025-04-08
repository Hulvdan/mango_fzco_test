from contextlib import asynccontextmanager
from datetime import datetime
from typing import TypeAlias

from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import delete, exists, func, select

from . import utils
from .db import (
    Chat,
    ChatParticipant,
    Group,
    Message,
    RecentlySucceededOperation,
    Unread,
    User,
)
from .utils import log

UserID: TypeAlias = int
MessageID: TypeAlias = str | None
ChatID: TypeAlias = int

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
        log.info("User with `%s` email doesn't exist!", data.email)
        raise UserDoesNotExistError

    if not _password_context.verify(data.password, user.password):
        log.info("User with `%s` email provided a wrong password!", data.email)
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
        log.info("User %d supplied too many participants for group creation!", creator_id)
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
        log.info(
            "User %d supplied non existent participants for group creation!", creator_id
        )
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

    # int32 id операции для предотвращения дублирования сообщений.
    # Пользователь бы, отправляя несколько запросов создания сообщения
    # разом в несколько сервисов, отправлял бы один и тот же рандомно сгенерированный
    # operation_id.
    #
    # Если бы один сервис уже создал сообщение,
    # пользователю временно нельзя было бы создать сообщение с этим operation_id.
    operation_id: int = Field(ge=utils.INT16_MIN, le=utils.INT16_MAX)


class MessageData(BaseModel):
    id: int
    chat_id: int
    sender_id: int
    text: str
    timestamp: datetime
    is_read: bool


class OperationAlreadySucceededError(DomainError):
    status = 400


async def check_not_duplicated(sender_id: int, operation_id: int, session) -> None:
    st = select(
        exists()
        .select_from(RecentlySucceededOperation)
        .where(
            RecentlySucceededOperation.user_id == sender_id,
            RecentlySucceededOperation.operation_id == operation_id,
        )
    )
    if (await session.execute(st)).scalar():
        log.info(
            "User %d have recently committed an operation %d", sender_id, operation_id
        )
        raise OperationAlreadySucceededError


async def message_user(
    *, data: MakeMessageData, sender_id: int, receiver_id: int, session
) -> MessageData:
    await check_not_duplicated(sender_id, data.operation_id, session)

    # см. Chat.user_ids
    user1 = sender_id
    user2 = receiver_id
    if user2 < user1:
        user1, user2 = user2, user1
    user_ids = user1 << 32 + user2

    existing_chat = (
        await session.execute(select(Chat).where(Chat.user_ids == user_ids))
    ).scalar()

    if not existing_chat:
        log.info("Creating a chat for users %d and %d...", sender_id, receiver_id)
        existing_chat = Chat(type=Chat.TYPE_PERSONAL, user_ids=user_ids)
        session.add(existing_chat)
        await session.flush()

    message = Message(
        chat_id=existing_chat.id,
        sender_id=sender_id,
        text=data.text,
    )
    session.add(message)

    operation = RecentlySucceededOperation(
        user_id=sender_id, operation_id=data.operation_id
    )
    session.add(operation)

    await session.flush()

    session.add(Unread(message_id=message.id, user_id=receiver_id))
    await session.commit()

    return MessageData(
        id=message.id,
        chat_id=message.chat_id,
        sender_id=message.sender_id,
        text=message.text,
        timestamp=message.timestamp,
        is_read=message.is_read,
    )


class GroupDoesNotExistError(DomainError):
    status = 404


async def message_group(
    *, data: MakeMessageData, sender_id: int, group_id: int, session
) -> MessageData:
    await check_not_duplicated(sender_id, data.operation_id, session)

    chat_id = (
        await session.execute(select(Group.chat_id).filter(Group.id == group_id))
    ).scalar()

    if chat_id is None:
        log.info("Group %d does not exist", group_id)
        raise GroupDoesNotExistError

    message = Message(
        chat_id=chat_id,
        sender_id=sender_id,
        text=data.text,
    )
    session.add(message)

    operation = RecentlySucceededOperation(
        user_id=sender_id, operation_id=data.operation_id
    )
    session.add(operation)

    await session.flush()

    readers = (
        await session.execute(
            select(ChatParticipant.user_id).where(ChatParticipant.chat_id == chat_id)
        )
    ).scalars()

    session.add_all(
        Unread(message_id=message.id, user_id=user_id)
        for user_id in readers
        if user_id != sender_id
    )

    await session.commit()

    return MessageData(
        id=message.id,
        chat_id=message.chat_id,
        sender_id=message.sender_id,
        text=message.text,
        timestamp=message.timestamp,
        is_read=message.is_read,
    )


ReadMessageData: TypeAlias = tuple[ChatID, UserID]


async def read_message(message_id: int, user_id: int, session) -> ReadMessageData | None:
    unread = (
        await session.execute(
            select(Unread)
            .where(Unread.message_id == message_id, Unread.user_id == user_id)
            .limit(1)
        )
    ).scalar()
    if not unread:
        return None

    await session.delete(unread)
    await session.flush()

    st = select(exists().select_from(Unread).where(Unread.message_id == message_id))
    is_read = not (await session.execute(st)).scalar()

    return_value: ReadMessageData | None = None

    if is_read:
        message = (
            await session.execute(select(Message).where(Message.id == message_id))
        ).scalar()
        message.is_read = True
        session.add(message)

        return_value = (message.chat_id, message.sender_id)

    await session.commit()

    return return_value


class UserCantAccessSpecifiedChatError(DomainError):
    status = 403


async def history(
    *, chat_id: int, user_id: int, later_that_message_id: int, limit: int, session
) -> list[MessageData]:
    if not await can_access_chat(chat_id=chat_id, user_id=user_id, session=session):
        log.info(
            "User %d tried accessing chat %d that he is not a participant of!",
            user_id,
            chat_id,
        )
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
        (await session.execute(statement.order_by(Message.id.asc()).limit(limit)))
        .scalars()
        .all()
    )

    return [
        MessageData(
            id=i.id,
            chat_id=i.chat_id,
            sender_id=i.sender_id,
            text=i.text,
            timestamp=i.timestamp,
            is_read=i.is_read,
        )
        for i in messages
    ]


async def can_access_chat(*, chat_id: int, user_id: int, session) -> bool:
    return (
        await session.execute(
            select(exists().select_from(ChatParticipant)).where(
                ChatParticipant.chat_id == chat_id, ChatParticipant.user_id == user_id
            )
        )
    ).scalar()


async def purge_old_message_operations(session) -> None:
    log.info("Purging old message operations...")

    st = delete(RecentlySucceededOperation).where(
        RecentlySucceededOperation.timestamp < datetime.utcnow()  # noqa: DTZ003
    )
    await session.execute(st)

    log.info("Purged old message operations!")


async def fill_db_with_initial_data():
    from .db import make_session, reinit_db_from_scratch

    log.info("Filling db with with initial data...")

    await reinit_db_from_scratch()

    async with asynccontextmanager(make_session)() as session:
        for i in range(1, 4):
            await make_user(
                email=f"test{i}@test.com",
                name=f"test{i}",
                password="test",
                session=session,
            )

    log.info("Filled db with with initial data!")
