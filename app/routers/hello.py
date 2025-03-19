import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from app.core.config import settings

router = APIRouter()


@router.get("/hello")
async def get_static_file():

    return JSONResponse(content={"message": "Hello, world!"})
