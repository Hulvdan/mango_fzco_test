from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Type

import jwt
from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import db, services
from .settings import settings


# Пересоздаю БД при поднятии сервиса. Это бы убрали потом, конечно же
@asynccontextmanager
async def app_lifespan(_):
    await services.fill_db_with_initial_data()
    yield


app = FastAPI(lifespan=app_lifespan)


@app.exception_handler(services.DomainException)
async def unicorn_exception_handler(_: Request, exc: Type[services.DomainException]):
    return JSONResponse(
        status_code=exc.status, content={"message": exc.__class__.__name__}
    )


JWT_ALGORITHM = "HS256"


def create_access_token(user_id: int) -> str:
    return jwt.encode(
        {
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        },
        settings.service_secret,
        algorithm=JWT_ALGORITHM,
    )


async def get_user_id(
    token: Annotated[str, Depends(OAuth2PasswordBearer(tokenUrl="login"))],
) -> int:
    try:
        payload = jwt.decode(token, settings.service_secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401)

    return payload.get("user_id")


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
    token = create_access_token(user_id)
    return Token(access_token=token)


class MakeGroupResponse(BaseModel):
    group_id: int


@app.post("/group/", status_code=201)
async def make_group_endpoint(
    body: services.MakeGroup,
    user_id=Depends(get_user_id),
    session=Depends(db.make_session),
) -> MakeGroupResponse:
    return MakeGroupResponse(group_id=services.make_group(body, user_id, session))


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
