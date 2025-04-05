import asyncio
import json
from contextlib import asynccontextmanager
from typing import Callable

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import NullPool
from starlette.testclient import WebSocketTestSession

from . import db, main, services

client = TestClient(main.app)


def request_as_factory(client_method) -> Callable[[...], Response]:
    def request_as(user_id: int, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer {}".format(
            main.create_access_token(user_id, "test_client_id")
        )

        return client_method(*args, **kwargs, headers=headers)

    return request_as


def websocket_as_factory(client_method) -> Callable[[...], WebSocketTestSession]:
    def request_as(user_id: int, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer {}".format(
            main.create_access_token(user_id, "test_client_id")
        )

        ws = client_method(*args, **kwargs, headers=headers)

        async def receive_json_async():
            result = await asyncio.wait_for(ws._send_rx.receive(), 0.2)  # noqa: SLF001
            return json.loads(result["text"])

        ws.receive_json_async = receive_json_async
        return ws

    return request_as


client.get_as = request_as_factory(client.get)
client.post_as = request_as_factory(client.post)
client.put_as = request_as_factory(client.put)
client.patch_as = request_as_factory(client.patch)
client.delete_as = request_as_factory(client.delete)
client.websocket_connect_as = websocket_as_factory(client.websocket_connect)


@pytest.fixture(scope="session", autouse=True)
async def init_engine_and_sessionmaker():
    db.init_engine_and_sessionmaker(custom_poolclass=NullPool)


@pytest.fixture
async def session():
    # Каждый тест с пустой БД.
    #
    # Сделано на скорую руку. На реальном проекте я бы после каждого теста
    # TRUNCATE-ил бы все таблицы. Это позволило бы использовать транзакции в тестах.
    #
    # Можно было бы просто BEGIN / ROLLBACK на каждый тест ставить,
    # но на деле этого мне не хватало.
    await db.reinit_db_from_scratch()

    async with asynccontextmanager(db.make_session)() as session:
        yield session


def is_ok(response: Response):
    ok = 200 <= response.status_code < 300
    assert ok, response.text


_last_created_user = 0


async def create_user(session) -> int:
    global _last_created_user
    _last_created_user += 1
    return await services.make_user(
        email=f"test{_last_created_user}@test.com",
        password="test",
        name="test_name",
        session=session,
    )


async def test_login(session):
    user = await create_user(session)
    response = client.post(
        "login", data={"username": f"test{user}@test.com", "password": "test"}
    )
    is_ok(response)


async def test_make_group(session):
    user1 = await create_user(session)
    user2 = await create_user(session)
    make_group_as(user1, [user1, user2])


async def make_group_as(user_id: int, participants: list[int]):
    response = client.post_as(
        user_id,
        "/group",
        json={"name": "test_group", "participant_user_ids": participants},
    )
    is_ok(response)
    j = response.json()
    return j["group_id"], j["chat_id"]


def message_group_as(user_id: int, group_id: int, text: str):
    response = client.post_as(
        user_id,
        f"message/group/{group_id}",
        json={"text": text},
    )
    is_ok(response)


async def test_group_messaging(session):
    user1 = await create_user(session)
    user2 = await create_user(session)
    user3 = await create_user(session)

    group_123, _ = await make_group_as(user2, [user1, user2, user3])
    group_23, _ = await make_group_as(user2, [user2, user3])

    ws = lambda user_id: client.websocket_connect_as(user_id, "/ws/")

    with ws(user1) as ws1, ws(user2) as ws2, ws(user3) as ws3:
        message_group_as(user2, group_123, "1")
        m1 = await ws1.receive_json_async()
        m3 = await ws3.receive_json_async()
        assert m1["text"] == "1"
        assert m3["text"] == "1"
        # user2 - автор сообщения, не отправляем ему.
        with pytest.raises(asyncio.TimeoutError):
            await ws2.receive_json_async()

        message_group_as(user2, group_23, "2")
        m3 = await ws3.receive_json_async()
        assert m3["text"] == "2"
        with pytest.raises(asyncio.TimeoutError):
            await ws1.receive_json_async()
        with pytest.raises(asyncio.TimeoutError):
            await ws2.receive_json_async()


async def test_cant_access_history(session):
    user = await create_user(session)
    response = client.get_as(user, "history/1")
    assert response.json()["message"] == "UserCantAccessSpecifiedChatError"


async def test_history(session):
    user = await create_user(session)
    group, chat_id = await make_group_as(user, [user])

    message_group_as(user, group, "1")
    message_group_as(user, group, "2")
    message_group_as(user, group, "3")

    response = client.get_as(user, f"history/{chat_id}")
    is_ok(response)

    j = response.json()
    assert j[0]["text"] == "3"
    assert j[1]["text"] == "2"
    assert j[2]["text"] == "1"

    response = client.get_as(
        user, f"history/{chat_id}", params={"earlier_that_message_id": j[0]["id"]}
    )
    is_ok(response)
    j = response.json()
    assert len(j) == 2
    assert j[0]["text"] == "2"
    assert j[1]["text"] == "1"
