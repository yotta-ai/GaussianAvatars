# app/services/video_service.py
import base64
import logging
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional

from app.core.config import config
from app.core.constants import (
    REQUIRED_FRAMES_FOR_PLAYBACK,
    FRAME_MESSAGE,
    AUDIO_MESSAGE,
)
from app.services.utils import LocalViewer
from gaussian_renderer import render


class VideoService:
    """Handles video frame generation and streaming."""

    def __init__(self):
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
        self.max_cache_size = 100  # Limit cache size

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

    def _generate_single_frame(self, frame_idx: int) -> Optional[Dict]:
        """Generates a single video frame."""
        if frame_idx >= len(self.visemes):
            return None

        try:
            torch.cuda.synchronize()

            current_frame = self.visemes[frame_idx]
            self.viewer.apply_blendshapes(current_frame)

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

            timestamp = current_frame["time"] / 1000.0
            is_last_frame = frame_idx >= len(self.visemes) - 1

            return {
                "frame": frame_base64,
                "timestamp": timestamp,
                "is_last": is_last_frame,
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
        try:
            await websocket.send_json(
                {
                    "type": AUDIO_MESSAGE,
                    "data": self.audio_data,
                    "duration": self.total_duration,
                }
            )

            await websocket.receive_text()  # Wait for "audio_ready"

            while self.is_streaming:
                frame_data = await self.get_next_frame()
                if frame_data is None:
                    self.is_streaming = False
                    break

                await websocket.send_json(
                    {
                        "type": FRAME_MESSAGE,
                        "data": frame_data["frame"],
                        "timestamp": frame_data["timestamp"],
                        "is_last": frame_data["is_last"],
                    }
                )

                if frame_data["is_last"]:
                    self.is_streaming = False
                    break
        except WebSocketDisconnect:
            logging.info("WebSocket disconnected during streaming")
        except Exception as e:
            logging.exception(f"Streaming error: {e}")
        finally:
            self.is_streaming = False
            torch.cuda.empty_cache()  # Ensure resources are cleaned up

    def cleanup(self):
        """Clean up resources, clear caches."""
        self.frame_cache.clear()
        torch.cuda.empty_cache()
        self.is_streaming = False
        self.frame_idx = 0
