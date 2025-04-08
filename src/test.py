from contextlib import asynccontextmanager
from random import randint
from typing import Callable

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import NullPool
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from . import db, main, services, utils


@pytest.fixture(scope="session", autouse=True)
async def init_engine_and_sessionmaker():
    db.init_engine_and_sessionmaker(custom_poolclass=NullPool)


def request_as_factory(client_method) -> Callable[[...], Response]:
    def request_as(user_id: int, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer {}".format(
            main.create_access_token(user_id, "test_client_id")
        )

        return client_method(*args, **kwargs, headers=headers)

    return request_as


class CantReceiveError(Exception):
    pass


def websocket_connect_as_factory(client_method) -> Callable[[...], WebSocketTestSession]:
    def websocket_connect_as(user_id: int, client_id: str | None, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer {}".format(
            main.create_access_token(user_id, client_id)
        )

        ws = client_method(*args, **kwargs, headers=headers)

        def receive_json_non_blocking():
            if ws._send_rx.statistics().current_buffer_used == 0:  # noqa: SLF001
                raise CantReceiveError
            return ws.receive_json()

        ws.receive_json_non_blocking = receive_json_non_blocking
        return ws

    return websocket_connect_as


client = TestClient(main.app)
client.get_as = request_as_factory(client.get)
client.post_as = request_as_factory(client.post)
client.put_as = request_as_factory(client.put)
client.patch_as = request_as_factory(client.patch)
client.delete_as = request_as_factory(client.delete)
client.websocket_connect_as = websocket_connect_as_factory(client.websocket_connect)


def ws(user_id: int, chat_id: int, *, client_id: str | None = "test_client_id"):
    return client.websocket_connect_as(user_id, client_id, f"/ws/{chat_id}/")


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
    await make_group_as(user1, [user1, user2])


async def make_group_as(user_id: int, participants: list[int]):
    response = client.post_as(
        user_id,
        "/group",
        json={
            "name": "test_group",
            "participant_user_ids": participants,
        },
    )
    is_ok(response)
    j = response.json()
    return j["group_id"], j["chat_id"]


def message_group_as(user_id: int, group_id: int, text: str):
    response = client.post_as(
        user_id,
        f"messages/group/{group_id}",
        json={
            "text": text,
            "operation_id": randint(utils.INT16_MIN, utils.INT16_MAX),
        },
    )
    is_ok(response)
    return response.json()


async def test_group_messaging(session):
    user1 = await create_user(session)
    user2 = await create_user(session)
    user3 = await create_user(session)

    group_123, chat_123 = await make_group_as(user2, [user1, user2, user3])
    group_23, chat_23 = await make_group_as(user2, [user2, user3])

    with ws(user1, chat_123) as ws1, ws(user2, chat_123) as ws2, ws(
        user3, chat_123
    ) as ws3:
        message_group_as(user2, group_123, "1")
        m1 = ws1.receive_json_non_blocking()["message"]
        m3 = ws3.receive_json_non_blocking()["message"]
        assert m1["text"] == "1"
        assert m3["text"] == "1"
        # user2 - автор сообщения, не отправляем ему.
        with pytest.raises(CantReceiveError):
            ws2.receive_json_non_blocking()["message"]

    with pytest.raises(WebSocketDisconnect):
        with ws(user1, chat_23):
            pass

    with ws(user2, chat_23) as ws2, ws(user3, chat_23) as ws3:
        message_group_as(user2, group_23, "2")
        m3 = ws3.receive_json_non_blocking()["message"]
        assert m3["text"] == "2"
        with pytest.raises(CantReceiveError):
            ws2.receive_json_non_blocking()["message"]


async def test_read(session):
    user1 = await create_user(session)
    user2 = await create_user(session)

    group, chat = await make_group_as(user2, [user1, user2])
    message_response = message_group_as(user1, group, "1")

    response = client.get_as(user2, f"history/{chat}")
    assert response.json()[0]["is_read"] is False

    with ws(user1, chat) as ws1:
        read_response = client.post_as(
            user2, "messages/read/{}".format(message_response["id"])
        )
        is_ok(read_response)

        response = client.get_as(user2, f"history/{chat}")
        assert response.json()[0]["is_read"] is True

        # Автор получает отбивку о том, что его сообщение прочитали
        m = ws1.receive_json_non_blocking()
        assert m["type"] == "Read"
        assert m["message_id"] == message_response["id"]


# «Реализовать механизм подключения к нескольким устройствам одновременно»
async def test_user_can_connect_using_several_clients(session):
    user1 = await create_user(session)
    user2 = await create_user(session)

    group_12, chat_12 = await make_group_as(user2, [user1, user2])

    with ws(user1, chat_12) as ws1, ws(
        user1, chat_12, client_id="another_client"
    ) as ws1_another:
        message_group_as(user2, group_12, "1")
        m1 = ws1.receive_json_non_blocking()["message"]
        m1_another = ws1_another.receive_json_non_blocking()["message"]
        assert m1["text"] == "1"
        assert m1_another["text"] == "1"


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
    assert j[0]["text"] == "1"
    assert j[1]["text"] == "2"
    assert j[2]["text"] == "3"

    response = client.get_as(
        user, f"history/{chat_id}", params={"later_that_message_id": j[0]["id"]}
    )
    is_ok(response)
    j = response.json()
    assert len(j) == 2
    assert j[0]["text"] == "2"
    assert j[1]["text"] == "3"
