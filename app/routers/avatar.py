import time
import json
import base64
import asyncio
from pathlib import Path
from fastapi import (
    FastAPI,
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from typing import Set, Any, Dict, Optional, Callable

from app.services import viseme_service
from app.services import video_service
from app.services import gcloud_service
from app.services import lifeguru_service
from app.services.openai_service import OpenAIService, AsyncRealtimeConnection
from app.services.lifeguru_service import LifeGuruService
from app.services.viseme_service import VisemeService
from app.services.video_service import VideoService
from app.services.monitoring_service import ConnectionMonitor
from app.services.gcloud_service import GcloudService
from app.core.config import OPENAI_API_KEY, Config
from app.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


class ConnectionManager:
    """Handles WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and store a WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            "WebSocket connected. Active connections: %d", len(self.active_connections)
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(
            "WebSocket disconnected. Active connections: %d",
            len(self.active_connections),
        )

    async def send_message(self, message: str, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket connection."""
        await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        """Send a message to all active WebSocket connections."""
        for connection in self.active_connections:
            await connection.send_text(message)

    async def check_connection(
        self,
        websocket: WebSocket,
        callback: Optional[Callable] = None,
        session_id=None,
        token=None,
    ):
        while True:
            if (
                hasattr(websocket, "client_state")
                and websocket.client_state.name != "CONNECTED"
            ):
                logger.info(
                    f"WebSocket client state is {websocket.client_state.name}, removing connection"
                )
                self.disconnect(websocket)
                logger.info("WebSocket disconnected for session_id=%s", session_id)
                await LifeGuruService().end_session(session_id=session_id, token=token)
                if callback:
                    await callback()
                break
            await asyncio.sleep(1)


class OpenAIStreamHandler:
    """Handles interaction with OpenAI's streaming API for a given WebSocket session."""

    def __init__(
        self,
        openai_service: OpenAIService,
        websocket: WebSocket,
        session_id: str,
        token: str,
        model_id: int,
    ):
        self.openai_service = openai_service
        self.websocket = websocket
        self.session_id = session_id
        self.token = token
        self.model_id = model_id
        self.ai_response_start_timestamp: int = 0
        self.latest_client_media_timestamp: int = 0
        self.last_assistant_item: Any = None
        self.tasks = []
        self.viseme_service: VisemeService = None
        self.video_service: VideoService = None

        self.frame_generation_task: Optional[asyncio.Task] = None
        self.should_cancel_frame_generation = False

    async def receive_from_client(
        self, openai_connection: AsyncRealtimeConnection
    ) -> None:
        """Receive messages from client and forward them to OpenAI."""
        try:
            async for message in self.websocket.iter_text():
                data = json.loads(message)
                event_type = data.get("event")
                if event_type == "start":
                    self._handle_start_event()
                elif event_type == "audio":
                    await self._handle_audio_event(data, openai_connection)
                elif event_type == "text":
                    print(data)
                    await self._handle_text_event(data, openai_connection)
        except WebSocketDisconnect:
            logger.info("Client disconnected (receive_from_client).")
        except Exception as e:
            logger.error("Unexpected error in receive_from_client: %s", e)

    async def send_to_client(self, openai_connection: Any) -> None:
        """Receive events from OpenAI and forward them to the client."""
        try:
            async for event in openai_connection:
                self.openai_service.handle_conversation_event(event)
                user_message = (
                    self.openai_service.last_user_message.text
                    if self.openai_service.last_user_message
                    and not self.openai_service.last_user_message.acknowledgement_status
                    else None
                )
                assistant_message = (
                    self.openai_service.last_assistant_message.text
                    if self.openai_service.last_assistant_message
                    and not self.openai_service.last_assistant_message.acknowledgement_status
                    else None
                )
                if user_message or assistant_message:
                    asyncio.create_task(
                        self._save_conversation(
                            user_message,
                            assistant_message,
                        )
                    )

                if assistant_message:
                    print("New Assistant Message Received")
                    print(self.openai_service.last_assistant_message.text)

                    # Cancel any existing frame generation task before starting a new one
                    await self._cancel_frame_generation()

                    # Start a new frame generation task asynchronously
                    self.should_cancel_frame_generation = False
                    self.frame_generation_task = asyncio.create_task(
                        self._process_and_stream_frames(
                            self.openai_service.last_assistant_message.text
                        )
                    )
                    self.tasks.append(self.frame_generation_task)
                if user_message:
                    self.openai_service.last_user_message.acknowledgement_status = True
                if assistant_message:
                    self.openai_service.last_assistant_message.acknowledgement_status = (
                        True
                    )

                if event.type == "response.text.delta":
                    await self._handle_text_response(event)
                elif event.type == "response.audio.delta":
                    await self._handle_audio_response(event)
                elif event.type == "input_audio_buffer.speech_started":
                    await self._interrupt_client()
                elif event.type == "response.done":
                    await self._handle_function_call(event, openai_connection)
                # Optionally handle other event types here
        except Exception as e:
            logger.error("Error in send_to_client: %s", e)

    async def _process_and_stream_frames(self, text: str) -> None:
        """Process the text to generate visemes and stream frames asynchronously."""
        try:
            # Generate visemes and audio
            response_data = self.viseme_service.generate_visemes_and_audio(
                text, self.model_id
            )

            # Check if the task should be cancelled
            if self.should_cancel_frame_generation:
                logger.info("Frame generation cancelled before processing visemes")
                return

            if response_data and response_data["visemes"] and response_data["audio"]:
                # Set visemes and audio
                interpolated_visemes = (
                    self.viseme_service.convert_visemes_to_flame_parameters(
                        response_data["visemes"]
                    )
                )
                self.video_service.set_visemes_and_audio(
                    interpolated_visemes, response_data["audio"]
                )

                # Check if the task should be cancelled again
                if self.should_cancel_frame_generation:
                    logger.info("Frame generation cancelled before streaming")
                    return

                # Stream the video
                await self.video_service.stream_video(self.websocket)
        except asyncio.CancelledError:
            logger.info("Frame generation task was cancelled")
        except Exception as e:
            logger.error(f"Error in process_and_stream_frames: {e}")

    async def _cancel_frame_generation(self) -> None:
        """Cancel the current frame generation task if it exists."""
        if self.frame_generation_task and not self.frame_generation_task.done():
            self.should_cancel_frame_generation = True
            try:
                # Signal to the video service to stop streaming
                self.video_service.is_streaming = False

                # Wait a short time for natural cancellation
                await asyncio.sleep(0.1)

                # If still not done, cancel the task
                if not self.frame_generation_task.done():
                    self.frame_generation_task.cancel()
                    try:
                        await self.frame_generation_task
                    except asyncio.CancelledError:
                        logger.info("Frame generation task cancelled")
            except Exception as e:
                logger.error(f"Error cancelling frame generation: {e}")

    def _handle_start_event(self) -> None:
        """Reset tracking variables for a new streaming session."""
        self.ai_response_start_timestamp = 0
        self.latest_client_media_timestamp = 0
        self.last_assistant_item = None
        logger.info("Incoming stream started for session_id=%s", self.session_id)

    async def _handle_audio_event(
        self, data: Dict[str, Any], openai_connection: AsyncRealtimeConnection
    ) -> None:
        """Forward client audio to OpenAI."""
        if openai_connection:
            # print("Audio Event Received",data["audio"]["payload"])
            self.latest_client_media_timestamp = int(data["audio"]["timestamp"] or 0)
            await openai_connection.input_audio_buffer.append(
                audio=data["audio"]["payload"]
            )

    async def _handle_text_event(
        self, data: Dict[str, Any], openai_connection: AsyncRealtimeConnection
    ) -> None:
        """Forward client text to OpenAI."""
        try:
            text = data["text"]["payload"]
            print("Text Event Received", text)
            await openai_connection.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": {
                            "type": "input_text",
                            "text": data["text"]["payload"],
                        },
                    },
                }
            )
            await openai_connection.response.create()
            print("Response created successfully!")
        except Exception as e:
            logger.error("Error sending text data: %s", e)

    async def _handle_audio_response(self, event: Any) -> None:
        """Process audio response from OpenAI and send it to the client."""
        try:
            audio_payload = base64.b64encode(base64.b64decode(event.delta)).decode(
                "utf-8"
            )
            if self.ai_response_start_timestamp == 0:
                self.ai_response_start_timestamp = self.latest_client_media_timestamp
            if hasattr(event, "item_id") and event.item_id:
                self.last_assistant_item = event.item_id
            audio_delta = {
                "event": "audio",
                "audio": {"payload": audio_payload, "item_id": event.item_id},
            }
            await self.websocket.send_json(audio_delta)
        except Exception as e:
            logger.error("Error processing audio data: %s", e)

    async def _handle_text_response(self, event: Any) -> None:
        """Process audio response from OpenAI and send it to the client."""
        try:
            if self.ai_response_start_timestamp == 0:
                self.ai_response_start_timestamp = self.latest_client_media_timestamp
            if hasattr(event, "item_id") and event.item_id:
                self.last_assistant_item = event.item_id
            # print(event)
            audio_delta = {
                "event": "text",
                "text": {"payload": event.delta, "item_id": event.item_id},
            }
            await self.websocket.send_json(audio_delta)
        except Exception as e:
            logger.error("Error processing audio data: %s", e)

    async def _interrupt_client(self) -> None:
        """Interrupt client response when speech is detected."""
        print("Speech Detected")
        if self.ai_response_start_timestamp:
            elapsed_time = (
                self.latest_client_media_timestamp - self.ai_response_start_timestamp
            )
            if self.last_assistant_item:
                truncate_event = {
                    "type": "conversation.item.truncate",
                    "item_id": self.last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": elapsed_time,
                }
                event = {
                    "event": "interrupt",
                }
                # Cancel frame generation on interrupt
                await self._cancel_frame_generation()
                await self.websocket.send_json(event)
            self.last_assistant_item = None
            self.ai_response_start_timestamp = 0

    async def _handle_function_call(
        self, event: Any, openai_connection: AsyncRealtimeConnection
    ) -> None:
        """Handle function calls from OpenAI."""
        response = event.response
        if response.output and response.output[0].type == "function_call":
            fn_name = response.output[0].name
            fn_args = response.output[0].arguments
            fn_call_id = response.output[0].call_id
            logger.info("Function called: %s", fn_name)
            if fn_name == "end_conversation":
                asyncio.create_task(self._end_conversation(5))
            elif fn_name == "schedule_google_calender_event":
                asyncio.create_task(
                    self._schedule_calender_event(
                        fn_args, fn_call_id, openai_connection
                    )
                )

    async def _save_conversation(self, user_message, assistant_message):
        if user_message or assistant_message:
            await LifeGuruService().save_message_to_backend(
                token=self.token,
                session_id=self.session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )

    async def _schedule_calender_event(
        self, fn_arguments: str, fn_id, openai_connection: AsyncRealtimeConnection
    ) -> None:
        """Schedule a Google Calendar event."""
        print("Scheduling Google Calendar Event", fn_arguments)
        fn_arguments = json.loads(fn_arguments)
        response = await LifeGuruService().schedule_google_calendar_event(
            fn_arguments.get("event_description"), self.token
        )
        await openai_connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": fn_id,
                "output": json.dumps(response),
            }
        )
        await openai_connection.response.create()

    async def _end_conversation(self, delay: int) -> None:
        """Cleanly end the conversation after a delay."""
        await asyncio.sleep(delay)
        logger.info("Ending conversation for session_id=%s.", self.session_id)
        for task in self.tasks:
            if not task.done():
                task.cancel()


manager = ConnectionManager()
gcloud_service = GcloudService()
connection_monitor = ConnectionMonitor(
    connection_manager=manager,
    inactivity_timeout=180,
    stop_callback=gcloud_service.stop_instance,
)


def register_connection_monitor(app: FastAPI):
    """Register the connection monitor to start on application startup."""

    @app.on_event("startup")
    async def startup_event():
        connection_monitor.start_monitoring()
        logger.info("Connection monitor started")

    @app.on_event("shutdown")
    async def shutdown_event():
        connection_monitor.stop_monitoring()
        logger.info("Connection monitor stopped")


@router.post("/increase-timeout")
async def increase_idle_timeout():
    """Increase the idle timeout for the connection monitor."""
    if connection_monitor.timer_task:
        connection_monitor.timer_task.cancel()
    if not connection_monitor.has_active_connections():
        connection_monitor.timer_task = asyncio.create_task(
            connection_monitor._timer_countdown()
        )


@router.websocket("/ws/avatar/{model_id}/{token}")
async def websocket_endpoint(
    websocket: WebSocket, token: str, model_id: int = 306
) -> None:
    """WebSocket endpoint for real-time communication."""
    session_id = ""
    try:
        print("Connection Request Received fro Token : ", token)
        session_data = await LifeGuruService().initialize_session(token=token)
        if not session_data:
            raise WebSocketException("Failed to initialize session")

        session_id = session_data.get("sessionId")
        prompt = session_data.get("prompt")
        prompt += """"
        Also there is one very important rule here.... your answer to user must not exceed 2 lines... try to answer in one line.
        Try very hard to answer in very short sentence as possible unless user explicitly asks to speak more. This is very important 
        that your answers are very very short and to the point.
        """
        openai_service = OpenAIService(
            openai_api_key=OPENAI_API_KEY,
            system_message=prompt,
            show_timing_math=False,
        )

        await manager.connect(websocket)
        await connection_monitor.connection_state_changed()
        viseme_service = VisemeService()
        config = Config(
            point_path=Path(f"models/{model_id}/point_cloud.ply"),
            save_folder=Path("output_frames"),
            fps=45,
            demo_mode=True,
        )
        video_service = VideoService(config=config)
        if video_service.cam is None:
            video_service.cam = video_service.viewer.prepare_camera()
            asyncio.create_task(video_service.generate_gif(model_id, websocket))
        async with openai_service.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        ) as openai_connection:
            await openai_service.initialize_session(openai_connection)
            await openai_service.add_tools(openai_connection)

            handler = OpenAIStreamHandler(
                openai_service, websocket, session_id, token, model_id
            )
            handler.viseme_service = viseme_service
            handler.video_service = video_service
            receive_task = asyncio.create_task(
                handler.receive_from_client(openai_connection)
            )
            send_task = asyncio.create_task(handler.send_to_client(openai_connection))
            handler.tasks.extend([receive_task, send_task])
            asyncio.create_task(
                manager.check_connection(
                    websocket,
                    connection_monitor.connection_state_changed,
                    session_id,
                    token,
                )
            )
            await asyncio.gather(receive_task, send_task)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await connection_monitor.connection_state_changed()
        logger.info("WebSocket disconnected for session_id=%s", session_id)
        await LifeGuruService().end_session(session_id=session_id, token=token)
        logger.info("WebSocket disconnected for session_id=%s", session_id)
    except Exception as e:
        manager.disconnect(websocket)
        await connection_monitor.connection_state_changed()
        logger.info("WebSocket disconnected for session_id=%s", session_id)
        await LifeGuruService().end_session(session_id=session_id, token=token)
        logger.error("Unexpected error in websocket_endpoint: %s", e)
