import pytest
from fastapi.testclient import TestClient

from . import db, main, services

client = TestClient(main.app)


@pytest.fixture
async def session():
    await db.reinit_db_from_scratch()  # Каждый тест с пустой БД.

    with db.make_session() as session:
        yield session


async def test_message(session):
    await services.make_user(
        email="test@test.com", password="test", name="test_name", session=session
    )
