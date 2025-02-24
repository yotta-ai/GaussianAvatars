import time
import json
import base64
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException
from typing import Set, Any, Dict

from app.services.openai_service import OpenAIService,AsyncRealtimeConnection
from app.services.lifeguru_service import LifeGuruService
from app.core.config import OPENAI_API_KEY
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


class OpenAIStreamHandler:
    """Handles interaction with OpenAI's streaming API for a given WebSocket session."""

    def __init__(
        self, openai_service: OpenAIService, websocket: WebSocket, session_id: str
    ):
        self.openai_service = openai_service
        self.websocket = websocket
        self.session_id = session_id
        self.ai_response_start_timestamp: int = 0
        self.latest_client_media_timestamp: int = 0
        self.last_assistant_item: Any = None
        self.tasks = []

    async def receive_from_client(self, openai_connection: AsyncRealtimeConnection) -> None:
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
                if event.type == "response.text.delta":
                    await self._handle_text_response(event)
                elif event.type == "response.audio.delta":
                    await self._handle_audio_response(event)
                elif event.type == "input_audio_buffer.speech_started":
                    await self._interrupt_client()
                elif event.type == "response.done":
                    await self._handle_function_call(event)
                # Optionally handle other event types here
        except Exception as e:
            logger.error("Error in send_to_client: %s", e)

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
            print("Audio Event Received",data["audio"]["payload"])
            self.latest_client_media_timestamp = int(data["audio"]["timestamp"] or 0)
            await openai_connection.input_audio_buffer.append(audio=data["audio"]["payload"])

    async def _handle_text_event(
        self, data: Dict[str, Any], openai_connection: Any
    ) -> None:
        """Forward client text to OpenAI."""
        await openai_connection.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "content": data["text"]["payload"]},
            }
        )
        await openai_connection.response.create()

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
            print(event)
            audio_delta = {
                "event": "text",
                "text": {"payload": event.delta, "item_id": event.item_id},
            }
            await self.websocket.send_json(audio_delta)
        except Exception as e:
            logger.error("Error processing audio data: %s", e)

    async def _interrupt_client(self) -> None:
        """Interrupt client response when speech is detected."""
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
                await self.websocket.send_json(truncate_event)
            self.last_assistant_item = None
            self.ai_response_start_timestamp = 0

    async def _handle_function_call(self, event: Any) -> None:
        """Handle function calls from OpenAI."""
        response = event.response
        if response.output and response.output[0].type == "function_call":
            fn_name = response.output[0].name
            logger.info("Function called: %s", fn_name)
            if fn_name == "end_conversation":
                asyncio.create_task(self._end_conversation(5))

    async def _end_conversation(self, delay: int) -> None:
        """Cleanly end the conversation after a delay."""
        await asyncio.sleep(delay)
        logger.info("Ending conversation for session_id=%s.", self.session_id)
        for task in self.tasks:
            if not task.done():
                task.cancel()


@router.websocket("/ws/avatar/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
    """WebSocket endpoint for real-time communication."""
    session_id = ""
    try:
        print("Connection Request Received fro Token : ", token)
        session_data = await LifeGuruService().initialize_session(token=token)
        if not session_data:
            raise WebSocketException("Failed to initialize session")

        session_id = session_data.get("sessionId")
        prompt = session_data.get("prompt")
        openai_service = OpenAIService(
            openai_api_key=OPENAI_API_KEY,
            system_message=prompt,
            show_timing_math=False,
        )
        manager = ConnectionManager()
        await manager.connect(websocket)

        async with openai_service.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        ) as openai_connection:
            await openai_service.initialize_session(openai_connection)
            await openai_service.add_end_conversation_tool(openai_connection)

            handler = OpenAIStreamHandler(openai_service, websocket, session_id)
            receive_task = asyncio.create_task(
                handler.receive_from_client(openai_connection)
            )
            send_task = asyncio.create_task(handler.send_to_client(openai_connection))
            handler.tasks.extend([receive_task, send_task])
            await asyncio.gather(receive_task, send_task)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected for session_id=%s", session_id)
    except Exception as e:
        logger.error("Unexpected error in websocket_endpoint: %s", e)
