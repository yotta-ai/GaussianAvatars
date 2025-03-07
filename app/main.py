# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ws_router, avatar, static
from app.routers.avatar import register_connection_monitor

app = FastAPI(title="Lipsync Video Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the websocket router.
app.include_router(ws_router.router)
app.include_router(avatar.router)
app.include_router(static.router)
register_connection_monitor(app)


@app.get("/")
async def health_check():
    return {"message": "WebSocket Lipsync Server Running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
