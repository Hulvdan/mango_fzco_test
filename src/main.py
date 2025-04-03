import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypeAlias

import jwt
from fastapi import Depends, FastAPI, Header, Path, Query, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import db, services
from .settings import settings


# Пересоздаю БД при поднятии сервиса. Это бы убрали потом, конечно же
@asynccontextmanager
async def app_lifespan(_):
    db.init_engine_and_sessionmaker()
    await services.fill_db_with_initial_data()
    yield


app = FastAPI(lifespan=app_lifespan)


@app.exception_handler(services.DomainError)
async def unicorn_exception_handler(_: Request, exc):
    return JSONResponse(
        status_code=exc.status, content={"message": exc.__class__.__name__}
    )


JWT_ALGORITHM = "HS256"


def create_access_token(user_id: int, client_id: str | None) -> str:
    return jwt.encode(
        {
            "user_id": user_id,
            "client_id": client_id,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        },
        settings.service_secret,
        algorithm=JWT_ALGORITHM,
    )


UserID: TypeAlias = int
ClientID: TypeAlias = str | None
ChatID: TypeAlias = int


async def get_user_and_client_id(
    token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="login"))],
) -> tuple[UserID, ClientID]:
    try:
        payload = jwt.decode(token, settings.service_secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise services.NotAuthorizedError

    return payload.get("user_id"), payload.get("client_id")


class Token(BaseModel):
    # Сюда бы потом добавили refresh_token
    access_token: str


@app.post("/login/")
async def token_endpoint(
    body: Annotated[OAuth2PasswordRequestForm, Depends()],
    session=Depends(db.make_session),
) -> Token:
    user_id = await services.login(
        services.Login(email=body.username, password=body.password), session
    )
    token = create_access_token(user_id, body.client_id)
    return Token(access_token=token)


class MakeGroupResponse(BaseModel):
    group_id: int


@app.post("/group/", status_code=201)
async def make_group_endpoint(
    body: services.MakeGroup,
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> MakeGroupResponse:
    group_id = await services.make_group(body, user_and_client_id[0], session)
    return MakeGroupResponse(group_id=group_id)


ConnectedUserClient: TypeAlias = tuple[int, ClientID]
GroupID: TypeAlias = int


connected_personal: dict[ConnectedUserClient, UserID] = {}
connected_groups: dict[GroupID, list[ConnectedUserClient]] = {}


connected_users: dict[UserID, WebSocket] = {}


@app.post("/message/user/{user_id}")
async def message_user_endpoint(
    data: services.MakeMessage,
    user_id: int = Path(),
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> None:
    pass


background_tasks = set()


@app.post("/message/group/{group_id}")
async def message_group_endpoint(
    data: services.MakeMessage,
    group_id: int = Path(),
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> services.MessageData:
    message, users_that_could_see = await services.message_group(
        data=data, sender_id=user_and_client_id[0], group_id=group_id, session=session
    )

    for user_id in users_that_could_see:
        if user_id == user_and_client_id[0]:
            continue

        ws = connected_users.get(user_id)
        if ws:
            task = asyncio.create_task(ws.send_text(message.json()))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    return message


@app.get("/history/{chat_id}")
async def history_endpoint(
    chat_id: int,
    user_id=Depends(get_user_and_client_id),
    earlier_that_message_id=Query(0),
    limit=Query(0, gt=0, le=100),
    session=Depends(db.make_session),
) -> list[services.MessageData]:
    return await services.history(
        chat_id=chat_id,
        user_id=user_id,
        earlier_that_message_id=earlier_that_message_id,
        limit=limit,
        session=session,
    )


class MessageData(BaseModel):
    chat_id: int
    text: str
    deduplication_key: str


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket, authorization: str = Header(...)):
    scheme, param = get_authorization_scheme_param(authorization)
    if not authorization or scheme.lower() != "bearer":
        await websocket.close()
        return

    try:
        user_and_client_id = await get_user_and_client_id(param)
    except services.DomainError:
        await websocket.close()
        return

    user_id = user_and_client_id[0]

    try:
        await websocket.accept()
        connected_users[user_id] = websocket
        await asyncio.Future()  # Бесконечно ждём...
    finally:
        connected_users.pop(user_id, None)
