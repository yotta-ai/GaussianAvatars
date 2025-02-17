# app/services/llm_service.py

import requests
import uuid
import asyncio 
import logging
from app.core.config import settings  # Import settings

class LLMService:
    def __init__(self):
        self.llm_api_url = settings.LLM_API_URL
        self.voice_to_text_api_url = settings.VOICE_TO_TEXT_API_URL
        self.get_session_identifier_api_url = settings.GET_SESSION_IDENTIFIER_API_URL
        self.api_auth_token = settings.API_AUTHORIZATION_TOKEN

    def get_session_identifier(self, token):
        """Retrieves a session identifier from the API."""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(self.get_session_identifier_api_url, headers=headers, timeout=60) #added timeout
            response.raise_for_status()  # Raises HTTPError for bad requests (4xx or 5xx)
            response_data = response.json()
            return response_data.get("identifier", str(uuid.uuid4()))
        except requests.RequestException as e:
            logging.error(f"[ERROR] Identifier API Error: {e}")
            return str(uuid.uuid4())  # Fallback to a random UUID

    def generate_response(self, query: str, identifier: str, audio_request_id: str, token: str) -> str:
        """Generates a response from the LLM using the given query and identifier."""
        try:
            data = {"query": query, "sessionId": identifier, "audio_request_id": audio_request_id}
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            response = requests.post(self.llm_api_url, json=data, headers=headers, stream=True, timeout=60)  # Added timeout
            response.raise_for_status()

            compiled_response = ""
            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    compiled_response += chunk + "\n"
                    # In a real application, you might send these chunks to the client via WebSocket
            return compiled_response

        except requests.RequestException as e:
            logging.error(f"API Request Error: {e}")
            return str(e)

    async def transcribe_voice(self, audio_blob):
        """Transcribes voice data to text."""
        try:
            form_data = {'file': audio_blob}
            headers = {'Authorization': self.api_auth_token}
            response = await asyncio.to_thread(
                requests.post,
                self.voice_to_text_api_url,
                files=form_data,
                headers=headers,
                timeout=60  # Add a timeout to prevent indefinite hanging
            )
            response.raise_for_status()
            data = response.json()
            if data.get('converted_text'):
                return data['converted_text'].get('text'), data.get('id')
            else:
                logging.error("[Voice] No transcription received")
                return None, None

        except requests.RequestException as e:
            logging.error(f"[Voice] Error sending voice data: {e}")
            return None, None