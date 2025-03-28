import time
import statistics
from elevenlabs import ElevenLabs
from app.core.config import settings
from io import BytesIO
from pydub import AudioSegment
from app.logger import setup_logger

logger = setup_logger(__name__)


class ElevenLabsService:
    """
    Service to interact with Eleven Labs TTS API.
    """

    def __init__(
        self,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_22050_32",  # "pcm_16000", #"mp3_44100_128",
    ):

        self.client = ElevenLabs(
            api_key=settings.ELEVENLABS_API_KEY,
        )
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format

    def tts(self, text, save_to_file: bool = False):
        chunks = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            output_format=self.output_format,
            text=text,
            model_id=self.model_id,
        )

        mp3_content = b"".join(
            chunks
        )  # Combine all chunks into a single MP3 byte stream

        # Convert MP3 to WAV
        mp3_audio = BytesIO(mp3_content)
        audio: AudioSegment = AudioSegment.from_file(mp3_audio, format="mp3")
        audio.set_frame_rate(16000)
        wav_buffer = BytesIO()
        audio_length = audio.__len__()
        logger.info(f"Audio length: {audio_length} seconds")
        audio.export(wav_buffer, format="wav")
        wav_content = wav_buffer.getvalue()

        if save_to_file:
            file_path = "output.wav"
            with open(file_path, "wb") as f:
                f.write(wav_content)
            return file_path

        return wav_content

    @staticmethod
    def wav_to_base64(wav_content: bytes) -> str:
        """
        Convert WAV content (bytes) to a Base64-encoded string.
        """
        import base64

        return base64.b64encode(wav_content).decode("utf-8")
