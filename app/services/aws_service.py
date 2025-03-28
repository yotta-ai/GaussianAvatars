import os
import sys
import boto3
import subprocess
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
from io import BytesIO
from pydub import AudioSegment

from app.core.config import settings


class AWSService:
    def __init__(self):
        self.aws_client = boto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET,
            region_name="us-east-1",
        ).client("polly")

    def tts(self, text, model_id=306, output_format="mp3", save_to_file=False):
        try:
            # Request speech synthesis
            if model_id == 306:
                voice = "Arthur"
            elif model_id == 104:
                voice = "Gregory"
            elif model_id == 165:
                voice = "Stephen"
            elif model_id == 302:
                voice = "Amy"
            response = self.aws_client.synthesize_speech(
                Text=text,
                Engine="neural",
                OutputFormat=output_format,
                VoiceId=voice,
                SampleRate="16000",
            )
        except (BotoCoreError, ClientError) as error:
            raise Exception(f"Error synthesizing speech: {error}")

        # Access the audio stream from the response
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                mp3_content = stream.read()

                # Convert MP3 to WAV
                mp3_audio = BytesIO(mp3_content)
                audio: AudioSegment = AudioSegment.from_file(mp3_audio, format="mp3")
                audio.set_frame_rate(16000)
                wav_buffer = BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_content = wav_buffer.getvalue()

                if save_to_file:
                    file_path = "output.wav"
                    with open(file_path, "wb") as f:
                        f.write(wav_content)
                    return file_path

                return wav_content
        else:
            raise Exception("The response didn't contain audio data")


# aws_service = AWSService()
# aws_service.tts("Hello, this is a test.", save_to_file=True)
