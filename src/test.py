from contextlib import asynccontextmanager
from typing import Callable

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from . import db, main, services

client = TestClient(main.app)


def request_as_factory(client_method) -> Callable[[...], Response]:
    def request_as(_: int, *args, **kwargs):
        return client_method(*args, **kwargs)

    return request_as


client.get_as = request_as_factory(client.get)
client.post_as = request_as_factory(client.post)
client.put_as = request_as_factory(client.put)
client.patch_as = request_as_factory(client.patch)
client.delete_as = request_as_factory(client.delete)


_db_session_context_manager = asynccontextmanager(db.make_session)


@pytest.fixture
async def session():
    await db.reinit_db_from_scratch()  # Каждый тест с пустой БД.

    async with _db_session_context_manager() as session:
        yield session


_last_created_user = 0


async def create_user(session) -> int:
    global _last_created_user
    user_id = await services.make_user(
        email=f"test{_last_created_user}@test.com",
        password="test",
        name="test_name",
        session=session,
    )
    _last_created_user += 1
    return user_id


def is_ok(response: Response):
    ok = 200 <= response.status_code < 300
    assert ok, response.text


async def test_login(session):
    user = await create_user(session)
    response = client.post(
        "login", data={"username": f"test{user}@test.com", "password": "test"}
    )
    is_ok(response)


# async def test_make_group(session):
#     user1 = await create_user(session)
#     user2 = await create_user(session)
#
#     response = client.post_as(
#         user1,
#         "/group",
#         json={"name": "aboba", "participant_user_ids": [user1, user2]},
#     )
#     is_ok(response)
