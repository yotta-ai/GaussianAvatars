from openai import AsyncOpenAI
import openai
from openai.types.beta.realtime.session import Session
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from app.core import config
from typing import List
import openai.types.beta.realtime as openai_event_type
import app.services.openai_types as conversation_types
from app.logger import setup_logger
import json

logger = setup_logger(__name__)


class OpenAIService:
    def __init__(self, openai_api_key, system_message, show_timing_math):
        """
        Initialize the OpenAIService with the provided parameters.

        :param openai_api_key: API key for OpenAI
        :param system_message: Initial system message for the AI
        :param show_timing_math: Flag to show timing calculations
        """
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.system_message = system_message
        self.show_timing_math = show_timing_math
        self.log_event_types = [
            "error",
            "response.content.done",
            "rate_limits.updated",
            "response.done",
            "input_audio_buffer.committed",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.speech_started",
            "session.created",
        ]
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.conversation: conversation_types.Conversation = None
        self.last_assistant_message: conversation_types.LastMessage = None
        self.last_user_message: conversation_types.LastMessage = None

    async def send_initial_conversation_item(self, conn: AsyncRealtimeConnection):
        """
        Send the initial conversation item so the AI talks first.

        :param conn: The OpenAI realtime connection
        """
        initial_conversation_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This message is from system , user has picked the call, start conversation."
                        ),
                    }
                ],
            },
        }
        await conn.send(initial_conversation_item)
        await conn.response.create()

    async def add_tools(self, conn: AsyncRealtimeConnection):
        payload = {
            "tools": [
                {
                    "type": "function",
                    "name": "end_conversation",
                    "description": "Call this when you think that conversation has ended, e.g user says goodbye, or you think talking is not useful anymore.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "delay": {
                                "type": "number",
                                "description": "The delay in seconds before the conversation ends.",
                            },
                            "goodbye_message": {
                                "type": "string",
                                "description": "Good bye message for the user.",
                            },
                        },
                        "required": ["delay", "goodbye_message"],
                    },
                }
            ],
            "tool_choice": "auto",
        }
        payload["tools"].append(
            {
                "type": "function",
                "name": "schedule_google_calender_event",
                "description": "Call this Tool when user wants to schedule an event in his Google calendar.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_description": {
                            "type": "string",
                            "description": "A brief description of the event to be scheduled.",
                        },
                    },
                    "required": ["event_description"],
                },
            }
        )
        await conn.session.update(session=payload)

    async def add_custom_tool(
        self,
        conn: AsyncRealtimeConnection,
        name: str,
        description: str,
        parameter_properties: dict,
        required: List["str"],
    ):

        payload = {
            "tools": [
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameter_properties,
                        "required": required,
                    },
                }
            ],
            "tool_choice": "auto",
        }
        await conn.session.update(session=payload)

    async def initialize_session(self, conn: AsyncRealtimeConnection):
        """
        Initialize the session with OpenAI with the specified settings.

        :param conn: The OpenAI realtime connection
        """
        await conn.session.update(
            session={
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "instructions": self.system_message,
                "modalities": ["text"],
                "temperature": 0.8,
                "input_audio_transcription": {"model": "whisper-1"},
            }
        )
        await self.send_initial_conversation_item(conn)

    def handle_conversation_event(self, event: openai_event_type.RealtimeServerEvent):
        # logger.info("Event : %s", event.type)
        if isinstance(
            event, openai_event_type.session_created_event.SessionCreatedEvent
        ):
            self.conversation = conversation_types.Conversation(
                session_id=event.session.id
            )
        elif (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.type == "message"
            and event.item.role == "system"
        ):
            self.handle_system_item(event)
        elif (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.type == "message"
            and event.item.role == "user"
        ):
            self.handle_user_item_created(event)
        elif isinstance(
            event,
            openai_event_type.ConversationItemInputAudioTranscriptionCompletedEvent,
        ):
            self.handle_user_audio_transcription(event)
        elif (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.type == "message"
            and event.item.role == "assistant"
        ):
            self.handle_assistant_message_item_created(event)
        elif (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.type == "function_call"
        ):
            self.handle_assistant_function_call_item_created(event)
        elif (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.type == "function_call_output"
        ):
            self.handle_assistant_function_call_output_item_created(event)
        elif isinstance(
            event, openai_event_type.response_done_event.ResponseDoneEvent
        ) and (
            event.response.status == "completed" or event.response.status == "cancelled"
        ):
            self.handle_assistant_response_done(event)

    def handle_system_item(self, event: openai_event_type.RealtimeServerEvent):
        conversation_item = conversation_types.SystemConversationItem(
            id=event.item.id,
            previous_id=event.previous_item_id,
            type=conversation_types.ConversationItemType.message,
            role=conversation_types.ConversationItemRole.system,
            text=event.item.content[0].text,
        )
        self.conversation.items.append(conversation_item)

    def handle_user_item_created(self, event: openai_event_type.RealtimeServerEvent):
        if (
            isinstance(
                event,
                openai_event_type.conversation_item_created_event.ConversationItemCreatedEvent,
            )
            and event.item.status == "completed"
            and event.item.content[0].type == "input_audio"
        ):
            conversation_item = conversation_types.UserConversationItem(
                id=event.item.id,
                previous_id=event.previous_item_id,
                type=conversation_types.ConversationItemType.message,
                role=conversation_types.ConversationItemRole.user,
            )
            self.conversation.items.append(conversation_item)

    def handle_user_audio_transcription(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item: conversation_types.UserConversationItem = next(
            (item for item in self.conversation.items if item.id == event.item_id),
            None,
        )
        if conversation_item:
            conversation_item.text = event.transcript
            self.last_user_message = conversation_types.LastMessage(
                text=conversation_item.text
            )

    def handle_assistant_message_item_created(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item = conversation_types.AssistantConversationMessageItem(
            id=event.item.id,
            previous_id=event.previous_item_id,
            type=conversation_types.ConversationItemType.message,
            role=conversation_types.ConversationItemRole.assistant,
        )
        self.conversation.items.append(conversation_item)

    def handle_assistant_function_call_item_created(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item = conversation_types.AssistantConversationFunctionCallItem(
            id=event.item.id,
            previous_id=event.previous_item_id,
            type=conversation_types.ConversationItemType.function_call,
            role=conversation_types.ConversationItemRole.assistant,
            call_id=event.item.call_id,
            name=event.item.name,
        )
        self.conversation.items.append(conversation_item)

    def handle_assistant_response_done(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        if not len(event.response.output) > 0:
            return
        if event.response.output[0].type == "message":
            self.handle_assistant_message_response_done(event)
        elif event.response.output[0].type == "function_call":
            self.handle_assistant_function_call_response_done(event)

    def handle_assistant_message_response_done(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item: conversation_types.AssistantConversationMessageItem = next(
            (
                item
                for item in self.conversation.items
                if item.id == event.response.output[0].id
            ),
            None,
        )
        if conversation_item and event.response.output[0].content[0].type == "audio":
            conversation_item.text = event.response.output[0].content[0].transcript
        elif conversation_item and event.response.output[0].content[0].type == "text":
            conversation_item.text = event.response.output[0].content[0].text
            self.last_assistant_message = conversation_types.LastMessage(
                text=conversation_item.text
            )

        conversation_item.usage = conversation_types.ConversationUsage(
            **json.loads(event.response.usage.json())
        )

    def handle_assistant_function_call_response_done(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item: conversation_types.AssistantConversationFunctionCallItem = (
            next(
                (
                    item
                    for item in self.conversation.items
                    if item.id == event.response.output[0].id
                ),
                None,
            )
        )
        if conversation_item:
            conversation_item.arguments = event.response.output[0].arguments

        conversation_item.usage = conversation_types.ConversationUsage(
            **json.loads(event.response.usage.json())
        )

    def handle_assistant_function_call_output_item_created(
        self, event: openai_event_type.RealtimeServerEvent
    ):
        conversation_item = (
            conversation_types.AssistantConversationFunctionCallOutputItem(
                id=event.item.id,
                previous_id=event.previous_item_id,
                type=conversation_types.ConversationItemType.function_call_output,
                role=conversation_types.ConversationItemRole.system,
                call_id=event.item.call_id,
                output=event.item.output,
            )
        )
        self.conversation.items.append(conversation_item)
