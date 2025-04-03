from contextlib import asynccontextmanager
from typing import Callable

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import NullPool

from . import db, main, services

client = TestClient(main.app)


def request_as_factory(client_method) -> Callable[[...], Response]:
    def request_as(user_id: int, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = "Bearer {}".format(main.create_access_token(user_id))

        return client_method(*args, **kwargs, headers=headers)

    return request_as


client.get_as = request_as_factory(client.get)
client.post_as = request_as_factory(client.post)
client.put_as = request_as_factory(client.put)
client.patch_as = request_as_factory(client.patch)
client.delete_as = request_as_factory(client.delete)


@pytest.fixture(scope="session", autouse=True)
async def init_engine_and_sessionmaker():
    db.init_engine_and_sessionmaker(custom_poolclass=NullPool)


@pytest.fixture
async def session():
    await db.reinit_db_from_scratch()  # Каждый тест с пустой БД.

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

    response = client.post_as(
        user1,
        "/group",
        json={"name": "aboba", "participant_user_ids": [user1, user2]},
    )
    is_ok(response)
