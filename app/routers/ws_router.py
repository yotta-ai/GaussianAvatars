# app/routers/ws_router.py

import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.constants import ERROR_MESSAGE
from app.services.lifeguru_service import LifeGuruService
from app.services.viseme_service import VisemeService
from app.services.video_service import VideoService

router = APIRouter()
lifeguru_service = LifeGuruService()
viseme_service = VisemeService()
video_generator = VideoService()


@router.websocket("/ws/test/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for handling real-time lipsync generation.
    """
    await websocket.accept()
    identifier = await lifeguru_service.get_session_identifier(token=token)

    if not identifier:
        await websocket.send_json(
            {
                "type": "authentication",
                "message": "Failed to authenticate the connection",
            }
        )
        return
    # Start continuous frame generation and sending in the background
    if video_generator.cam is None:
        video_generator.cam = video_generator.viewer.prepare_camera()
    asyncio.create_task(video_generator.generate_gif(websocket))
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            request_data = data.get("data")
            text = request_data.get("text")
            audio_request_id = request_data.get("id", None)
            print(text)
            if text:
                logging.info(f"[INFO] Received text: {text}")
                # llm_response = await lifeguru_service.generate_response(text, identifier, audio_request_id, token)

                # Get visemes and audio from API
                response_data = viseme_service.get_visemes_and_audio_from_text(text)

                if (
                    response_data
                    and response_data["visemes"]
                    and response_data["audio"]
                ):
                    # Set visemes and audio, then stream the video.
                    interpolated_visemes = (
                        viseme_service.convert_visemes_to_flame_parameters(
                            response_data["visemes"]
                        )
                    )
                    video_generator.set_visemes_and_audio(
                        interpolated_visemes, response_data["audio"]
                    )
                    await video_generator.stream_video(websocket)
                else:
                    await websocket.send_json(
                        {
                            "type": ERROR_MESSAGE,
                            "message": "Failed to get visemes or audio",
                        }
                    )
            else:
                await websocket.send_json(
                    {"type": ERROR_MESSAGE, "message": "Invalid input data"}
                )

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.exception(f"[ERROR] WebSocket Error: {e}")
    finally:
        video_generator.cleanup()  # Important: clean up resources
