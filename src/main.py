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


# Пересоздаю БД при поднятии сервиса. Это бы убрали потом, конечно же.
@asynccontextmanager
async def app_lifespan(_):
    db.init_engine_and_sessionmaker()
    await services.fill_db_with_initial_data()
    yield


app = FastAPI(lifespan=app_lifespan)


@app.exception_handler(services.DomainError)
async def domain_errors_handler(_: Request, exc):
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
        services.LoginData(email=body.username, password=body.password), session
    )
    token = create_access_token(user_id, body.client_id)
    return Token(access_token=token)


@app.post("/group/", status_code=201)
async def make_group_endpoint(
    body: services.MakeGroupData,
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> services.MakeGroupResponse:
    return await services.make_group(body, user_and_client_id[0], session)


ConnectedUserClient: TypeAlias = tuple[int, ClientID]
GroupID: TypeAlias = int


# Пользователи могут подключиться с разных устройств
# Считаю, что устройство - client_id, который можно указать при логине.
connected_users: dict[UserID, list[tuple[ClientID, WebSocket]]] = {}


@app.post("/message/user/{user_id}")
async def message_user_endpoint(
    data: services.MakeMessageData,
    user_id: Annotated[int, Path()],
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> None:
    pass


background_tasks = set()


@app.post("/message/group/{group_id}")
async def message_group_endpoint(
    data: services.MakeMessageData,
    group_id: Annotated[int, Path()],
    user_and_client_id=Depends(get_user_and_client_id),
    session=Depends(db.make_session),
) -> services.MessageData:
    message, users_that_could_see = await services.message_group(
        data=data, sender_id=user_and_client_id[0], group_id=group_id, session=session
    )

    for user_id in users_that_could_see:
        if user_id == user_and_client_id[0]:
            continue

        for _, ws in connected_users.get(user_id, ()):
            task = asyncio.create_task(ws.send_text(message.json()))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    return message


@app.get("/history/{chat_id}")
async def history_endpoint(
    chat_id: int,
    user_and_client_id=Depends(get_user_and_client_id),
    # Фильтрование по Message.id > X не требует
    # от БД хождения по страницам, как OFFSET.
    # Также я не совсем понимаю, почему по возрастанию времени отправки бы сортировали, но ладно.
    later_that_message_id: int | None = Query(None),
    limit=Query(10, gt=0, le=100),
    session=Depends(db.make_session),
) -> list[services.MessageData]:
    return await services.history(
        chat_id=chat_id,
        user_id=user_and_client_id[0],
        later_that_message_id=later_that_message_id,
        limit=limit,
        session=session,
    )


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

    user_id, client_id = user_and_client_id

    try:
        await websocket.accept()

        clients = connected_users.get(user_id)
        if clients is None:
            clients = []
            connected_users[user_id] = clients

        # Кешируем подключение, пока не разорвётся соединение.

        # По-идее тут нужна была бы валидация, что этот пользователь с этим client_id
        # (устройством) не подключен к нам. Полировать можно бесконечно.
        clients.append((client_id, websocket))
        await asyncio.Future()

    finally:
        # Удаляю подключение без сохранения порядка в массиве,
        # чтобы не происходили лишние копирования.
        # Это бы называлось unstable remove / swap remove / remove with swap.
        clients = connected_users[user_id]
        for i in range(len(clients)):
            if clients[i][0] == client_id:
                clients[i] = clients[-1]
                clients.pop()
                if not clients:
                    del connected_users[user_id]
                break
