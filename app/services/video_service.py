import os
import time
import json
import random
import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

from pydantic import WebsocketUrl
import torch
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional

from app.core.config import settings
from app.core.constants import (
    REQUIRED_FRAMES_FOR_PLAYBACK,
    FRAME_MESSAGE,
    AUDIO_MESSAGE,
)
from app.services.utils import LocalViewer
from app.core.constants import FLAME_PARAMETERS, VISEME_MAPPING
from app.services.viseme_service import VisemeService
from app.core.constants import GIF_PATH
from gaussian_renderer import render


class VideoService:
    """Handles video frame generation and streaming."""

    def __init__(self, config):
        self.cfg = config
        self.viewer = LocalViewer(self.cfg)
        self.visemes = []
        self.frame_idx = 0
        self.is_streaming = False
        self.audio_data = None
        self.total_duration = 0
        self.cam = None
        self.background_color = torch.tensor(self.cfg.background_color).cuda()
        self.frame_cache = {}
        self.random_frames = []
        # self.max_cache_size = 100  # Limit cache size
        self.is_generating_random_frames = False
        self.frame_rate = self.cfg.fps
        self.viseme_service = VisemeService()
        self._stream_lock = asyncio.Lock()

    def set_visemes_and_audio(self, visemes, audio_base64):
        """Sets the viseme data and audio data for frame generation."""
        try:
            # Clear previous cache and reset states
            self.frame_cache.clear()
            torch.cuda.empty_cache()  # Clear CUDA cache

            self.visemes = visemes
            self.audio_data = audio_base64
            self.frame_idx = 0
            self.is_streaming = True

            if self.visemes:
                self.total_duration = max(v["time"] for v in self.visemes) / 1000.0
                print(f"Total duration: {self.total_duration} seconds")

            # Initialize camera if not already done
            if self.cam is None:
                self.cam = self.viewer.prepare_camera()

        except Exception as e:
            logging.error(f"Error in set_visemes_and_audio: {e}")
            self.is_streaming = False

    def generate_frame(self, frame_param_dict=None) -> Optional[Dict]:
        """Generates a single video frame."""
        try:
            torch.cuda.synchronize()
            # print("Frame Param Dict : ", frame_param_dict)
            self.viewer.apply_blendshapes(frame_param_dict.get("parameters"))

            with torch.cuda.device(self.background_color.device):
                render_output = render(
                    self.cam,
                    self.viewer.gaussians,
                    self.cfg.pipeline,
                    self.background_color,
                )

                frame_tensor = render_output.get("render")
                if frame_tensor is None:
                    return None

                with torch.no_grad():
                    frame_tensor = frame_tensor.detach()
                    frame_image = (torch.clamp(frame_tensor, 0, 1) * 255).to(
                        torch.uint8
                    )
                    frame_image = frame_image.permute(1, 2, 0).cpu().numpy()

            img = Image.fromarray(frame_image)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return img

        except Exception as e:
            logging.exception(f"Frame generation error: {e}")
            torch.cuda.empty_cache()
            return None

    def _generate_single_frame(self, frame_param_dict=None) -> Optional[Dict]:
        """Generates a single video frame."""
        try:
            torch.cuda.synchronize()
            # print("Frame Param Dict : ", frame_param_dict)
            self.viewer.apply_blendshapes(frame_param_dict.get("parameters"))

            with torch.cuda.device(self.background_color.device):
                render_output = render(
                    self.cam,
                    self.viewer.gaussians,
                    self.cfg.pipeline,
                    self.background_color,
                )

                frame_tensor = render_output.get("render")
                if frame_tensor is None:
                    return None

                with torch.no_grad():
                    frame_tensor = frame_tensor.detach()
                    frame_image = (torch.clamp(frame_tensor, 0, 1) * 255).to(
                        torch.uint8
                    )
                    frame_image = frame_image.permute(1, 2, 0).cpu().numpy()

            img = Image.fromarray(frame_image)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            timestamp = frame_param_dict["time"] / 1000.0

            return {
                "frame": frame_base64,
                "timestamp": timestamp,
                "is_last": False,
            }

        except Exception as e:
            logging.exception(f"Frame generation error: {e}")
            torch.cuda.empty_cache()
            return None

    async def get_next_frame(self) -> Optional[Dict]:
        """Gets the next frame, either from cache or by generating it."""
        if not self.visemes or self.frame_idx >= len(self.visemes):
            return None

        try:
            # if self.frame_idx in self.frame_cache:
            #     frame_data = self.frame_cache.pop(self.frame_idx)  # Remove from cache
            #     self.frame_idx += 1
            #     return frame_data

            frame_data = self._generate_single_frame(self.frame_idx)
            self.frame_idx += 1

            # if (
            #     self.frame_idx < len(self.visemes)
            #     and len(self.frame_cache) < self.max_cache_size
            # ):
            #     next_frame = self._generate_single_frame(self.frame_idx)
            #     if next_frame:
            #         self.frame_cache[self.frame_idx] = next_frame

            return frame_data

        except Exception as e:
            logging.exception(f"Error in get_next_frame: {e}")
            return None

    async def stream_video(self, websocket: WebSocket):
        """Streams video frames and audio to the client."""
        async with self._stream_lock:
            try:
                await websocket.send_json(
                    {
                        "type": AUDIO_MESSAGE,
                        "data": self.audio_data,
                        "duration": self.total_duration,
                    }
                )
                # Check streaming flag before starting
                if not self.is_streaming:
                    return

                frame_index = 0
                total_frames = len(self.visemes)

                # Process frames one by one with periodic cancellation checks
                while self.is_streaming and frame_index < total_frames:
                    # Check cancellation flag periodically
                    if frame_index % 5 == 0:  # Check every 5 frames
                        await asyncio.sleep(0)  # Yield to event loop
                        if not self.is_streaming:
                            logging.info("Frame streaming was cancelled")
                            break

                    frame_param = self.visemes[frame_index]
                    generated_frame = self._generate_single_frame(frame_param)
                    if generated_frame is None:
                        self.is_streaming = False
                        break

                    # Check if streaming should continue before sending
                    if not self.is_streaming:
                        break

                    await websocket.send_json(
                        {
                            "type": FRAME_MESSAGE,
                            "data": generated_frame["frame"],
                            "timestamp": generated_frame["timestamp"],
                            "is_last": (
                                True if frame_index + 1 == len(self.visemes) else False
                            ),
                        }
                    )
                    if frame_index + 1 == len(self.visemes):
                        self.is_streaming = False
                        break
                    frame_index += 1
            except WebSocketDisconnect:
                logging.info("WebSocket disconnected during streaming")
            except Exception as e:
                logging.exception(f"Streaming error: {e}")
            finally:
                self.is_streaming = False
                torch.cuda.empty_cache()  # Ensure resources are cleaned up

    def stop_streaming(self):
        """Safely stop the streaming process."""
        self.is_streaming = False
        logging.info("Video streaming stopped")

    def cleanup(self):
        """Clean up resources, clear caches."""
        self.frame_cache.clear()
        torch.cuda.empty_cache()
        self.is_streaming = False
        self.frame_idx = 0
        self.is_generating_random_frames = False  # Reset when a session is done

    async def get_idle_params(self):
        # read json list from file random_frames.json
        with open(
            "/home/yottacom/Projects/GaussianAvatars/idle_params.json", "r"
        ) as file:
            random_frames = json.load(file)
        return random_frames

    async def generate_gif(self,model_id, websocket: WebSocket):
        """ "Generate Gif of the currently loaded avatar"""

        gif_filename = f"idle_avatar_{model_id}.gif"
        gif_file_path = os.path.join(settings.BASE_DIR, "static", gif_filename)
        if os.path.exists(gif_file_path):
            print(f"File {gif_filename} already exists")
            await websocket.send_json(
                {
                    "type": GIF_PATH,
                    "path": gif_filename,
                }
            )
            return
        idle_params = await self.get_idle_params()
        interpolated_params = self.viseme_service.interpolate_blendshapes(
            idle_params, 10
        )
        iteration_count = 0
        idle_frames = []
        while True:
            for current_frame_index in range(len(interpolated_params)):
                current_frame_param = interpolated_params[current_frame_index]
                generated_frame = self.generate_frame(current_frame_param)
                idle_frames.append(generated_frame)
            iteration_count += 1
            if iteration_count == 3:
                break
        # Ensure we have frames before proceeding
        if idle_frames:
            # Set duration to 100ms per frame for 10 fps
            duration = 100  # milliseconds
            # Save the first frame and append the rest to create a GIF
            idle_frames[0].save(
                gif_file_path,
                format="GIF",
                save_all=True,
                append_images=idle_frames[1:],
                duration=duration,
                loop=0,  # 0 means the GIF will loop indefinitely
            )
            print(f"GIF saved as {gif_filename}")
            # Send the GIF to the client
            await websocket.send_json(
                {
                    "type": GIF_PATH,
                    "path": gif_filename,
                }
            )
