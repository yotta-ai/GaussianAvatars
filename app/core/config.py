# app/core/config.py
from pydantic import BaseModel, conset
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from pathlib import Path
from decouple import config as env_config


class Settings(BaseModel):
    """Configuration settings for the application."""

    LIFEGURU_API_URL: str = "https://api.lifeguruai.com/api"
    TTS_AVATAR_API_URL: str = "https://api.lifeguruai.com/api/TTS_Avatar"

    # WebSocket configurations.
    WEBSOCKET_HOST: str = "0.0.0.0"
    WEBSOCKET_PORT: int = 8001

    # Debug/Development options
    DEBUG_MODE: bool = False
    # get base directory of project by using Path(__file__).parent
    BASE_DIR: Path = Path(__file__).parent.parent.parent

    AWS_ACCESS_KEY: str = env_config("AWS_ACCESS_KEY")
    AWS_SECRET: str = env_config("AWS_SECRET")

    GCLOUD_AVATAR_INSTANCE_PROJECT: str = env_config("GCLOUD_AVATAR_INSTANCE_PROJECT")
    GCLOUD_AVATAR_INSTANCE_NAME: str = env_config("GCLOUD_AVATAR_INSTANCE_NAME")
    GCLOUD_AVATAR_INSTANCE_ZONE: str = env_config("GCLOUD_AVATAR_INSTANCE_ZONE")

    ELEVENLABS_API_KEY:str = env_config("ELEVENLABS_API_KEY")
    DEEPTALK_API_URL: str = env_config("DEEPTALK_API_URL")


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
print("Base directory ", settings.BASE_DIR)
OPENAI_API_KEY = env_config("OPENAI_API_KEY")
