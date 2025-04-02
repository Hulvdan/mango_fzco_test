from fastapi import FastAPI

from . import db

app = FastAPI()


@app.on_event("startup")
async def init_db():
    await db.init_db()


@app.get("/")
async def root():
    return {"message": "Hello World"}
