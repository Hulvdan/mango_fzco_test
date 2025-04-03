from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response, WebSocket

from . import db, services


@asynccontextmanager
async def app_lifespan(_):
    await db.reinit_db_from_scratch()
    yield


app = FastAPI(lifespan=app_lifespan)


@app.post("/group/")
async def make_group(
    body: services.MakeGroup,
    # user_id=TODO(),
    session=Depends(db.make_session),
):
    services.make_group(body, session)
    return Response(status_code=201)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
