import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings

router = APIRouter()


@router.get("/static/{file_name}")
async def get_static_file(file_name: str):
    file_path = os.path.join(settings.BASE_DIR, "static", file_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)
