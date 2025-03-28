import httpx
from app.core.config import settings


class DEEPTalkService:
    def __init__(self):
        self.api_url = settings.DEEPTALK_API_URL

    async def get_flame_params(self, audio_content: bytes):
        url = f"{self.api_url}/audio/flame-params"

        # Prepare the file payload using audio_content
        files = {
            "audio_file": ("audio.wav", audio_content, "audio/wav"),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files)

        # Handle the response
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
