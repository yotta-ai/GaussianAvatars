# app/core/config.py
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from pathlib import Path


class Settings(BaseModel):
    """Configuration settings for the application."""

    # LLM (Large Language Model) related configurations.
    LLM_API_URL: str = "https://api.lifeguruai.com/api/ConversationWithTrigger"
    VOICE_TO_TEXT_API_URL: str = "https://api.lifeguruai.com/api/VoiceToTxt"
    TTS_AVATAR_API_URL: str = "https://api.lifeguruai.com/api/TTS_Avatar"
    GET_SESSION_IDENTIFIER_API_URL: str = (
        "https://api.lifeguruai.com/api/GetSessionIdentifier"
    )
    API_AUTHORIZATION_TOKEN: str = (
        "Bearer 875d688653a74867891e2037d855cfd607df6bd9"  # Consider environment variables or secrets management
    )

    # WebSocket configurations.
    WEBSOCKET_HOST: str = "0.0.0.0"
    WEBSOCKET_PORT: int = 8001
    WEBSOCKET_URL: str = (
        "wss://rlsqs5jvkjuaac-8001.proxy.runpod.net/ws/875d688653a74867891e2037d855cfd607df6bd9"  # Consider dynamically building this
    )

    # Debug/Development options
    DEBUG_MODE: bool = False


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    point_path: Optional[Path] = None
    motion_path: Optional[Path] = None
    sh_degree: int = 3
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    save_folder: Path = Path("./viewer_output")
    fps: int = 25
    keyframe_interval: int = 1
    ref_json: Optional[Path] = None
    demo_mode: bool = False


config = Config(
    point_path=Path("media/306/point_cloud.ply"),
    save_folder=Path("output_frames"),
    fps=45,
    demo_mode=True,
)
settings = Settings()
