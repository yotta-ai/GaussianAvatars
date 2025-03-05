import httpx
import uuid
import asyncio
import logging
from app.core.config import settings  # Import settings


class LifeGuruService:
    def __init__(self):
        self.api_url = settings.LIFEGURU_API_URL

    async def get_session_identifier(self, token: str) -> str:
        """Retrieves a session identifier from the API asynchronously."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_url}/GetSessionIdentifier", headers=headers
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data.get("identifier", str(uuid.uuid4()))
        except httpx.RequestError as e:
            logging.error(f"[ERROR] Identifier API Error: {e}")
            return str(uuid.uuid4())  # Fallback to a random UUID

    async def initialize_session(self, token: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(
                    f"{self.api_url}/BuildPrompt", headers=headers
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data
        except httpx.RequestError as e:
            logging.error(f"[ERROR] Prompt API Error: {e}")
            return None

    async def generate_response(
        self, query: str, identifier: str, audio_request_id: str, token: str
    ) -> str:
        """Generates a response from the LLM using the given query and identifier asynchronously."""
        try:
            data = {
                "query": query,
                "sessionId": identifier,
                "audio_request_id": audio_request_id,
            }
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_url}/ConversationWithTrigger",
                    json=data,
                    headers=headers,
                )
                response.raise_for_status()

                compiled_response = ""
                async for chunk in response.aiter_lines():
                    if chunk:
                        compiled_response += chunk + "\n"
                return compiled_response
        except httpx.RequestError as e:
            logging.error(f"API Request Error: {e}")
            return str(e)

    async def transcribe_voice(self, token: str, audio_blob) -> tuple:
        """Transcribes voice data to text asynchronously."""
        try:
            form_data = {"file": ("audio.wav", audio_blob, "audio/wav")}
            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_url}/VoiceToTxt",
                    files=form_data,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                if data.get("converted_text"):
                    return data["converted_text"].get("text"), data.get("id")
                else:
                    logging.error("[Voice] No transcription received")
                    return None, None
        except httpx.RequestError as e:
            logging.error(f"[Voice] Error sending voice data: {e}")
            return None, None

    async def save_message_to_backend(
        self, token: str, session_id: str, user_message: str, assistant_message: str
    ) -> bool:
        message_obj = {
            "session_id": session_id,
            "user_message": user_message or "",
            "assistant_message": assistant_message or "",
        }
        url = f"{self.api_url}/saveMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            # Using httpx.AsyncClient for async request
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=message_obj)

                # Check if request was successful (status codes 200-299)
                if not (200 <= response.status_code < 300):
                    print(
                        f"Failed to save message: {response.status_code} - {response.text}"
                    )
                    return False

                print("Message saved successfully:", message_obj)
                return True

        except Exception as error:
            print(f"Error saving message to backend: {str(error)}")
            return False

    async def schedule_google_calendar_event(
        self, event_description: str, token: str
    ) -> dict:
        url = f"{self.api_url}/GoogleCalendarEvent"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"event_description": event_description}
        print("Scheduling event:", payload)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)

                if not (200 <= response.status_code < 300):
                    print(
                        f"Failed to schedule event: {response.status_code} - {response.text}"
                    )
                    return {
                        "success": False,
                        "message": "Failed to schedule the event.",
                    }

                result = response.json()
                print("Event scheduled successfully:", result)
                return {
                    "success": True,
                    "message": "Event scheduled successfully.",
                    "data": result,
                }

        except Exception as error:
            print(f"Error scheduling event: {str(error)}")
            return {
                "success": False,
                "message": "An error occurred while scheduling the event.",
            }
