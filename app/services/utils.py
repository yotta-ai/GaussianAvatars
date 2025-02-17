# app/services/utils.py
# This can contain shared utility functions, for example, your LocalViewer class.

# Copy utils from your original server.py and put here....
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel, render
from mesh_renderer import NVDiffRenderer


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


class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keyframes = []
        self.all_frames = {}
        self.num_record_timeline = 0
        self.playing = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians.binding is not None:
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)

        if self.gaussians.binding is not None:
            print("Initializing FLAME parameters...")
            self.reset_flame_param()

        super().__init__(cfg, "GaussianAvatars - Local Viewer")

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps

    def init_gaussians(self):
        """Load Gaussian Model."""
        if (self.cfg.point_path.parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        if self.cfg.point_path and self.cfg.point_path.exists():
            self.gaussians.load_ply(
                self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path
            )
        else:
            raise FileNotFoundError(f"{self.cfg.point_path} does not exist.")

    def reset_flame_param(self):
        """Reset FLAME blendshape parameters."""
        self.flame_param = {
            "expr": torch.zeros(1, self.gaussians.n_expr),
            "rotation": torch.zeros(1, 3),
            "neck": torch.zeros(1, 3),
            "jaw": torch.zeros(1, 3),
            "eyes": torch.zeros(1, 6),
            "translation": torch.zeros(1, 3),
        }

    def apply_blendshapes(self, blendshapes):
        """Properly handle FLAME model dimensions"""
        # Get actual model dimensions
        model_dims = {
            "expr": self.gaussians.n_expr,  # Get actual expression size dynamically
            "jaw": 3,
            "neck": 3,
            "eyes": 6,
            "rotation": 3,
            "translation": 3,
        }

        for key, values in blendshapes.items():
            if key not in self.flame_param:
                continue

            # Convert to tensor and ensure correct dimensions
            values = torch.tensor(values, dtype=torch.float32)

            # Ensure expression parameters match model dimension
            if key == "expr":
                if len(values) < model_dims["expr"]:
                    values = torch.cat(
                        [values, torch.zeros(model_dims["expr"] - len(values))]
                    )
                elif len(values) > model_dims["expr"]:
                    values = values[: model_dims["expr"]]

            # Ensure 6 eye parameters (duplicate if only 3 provided)
            elif key == "eyes":
                if len(values) == 3:
                    values = values.repeat(2)  # Duplicate for left and right eye
                values = values[:6]  # Ensure it doesn't exceed 6

            # Apply clamping based on parameter type
            if key in ["jaw", "neck", "eyes", "rotation", "translation"]:
                values = torch.clamp(values, -0.5, 0.5)
            elif key == "expr":
                values = torch.clamp(values, -3.0, 3.0)

            # Assign corrected values
            self.flame_param[key][0] = values[: model_dims.get(key, len(values))]

        self.gaussians.update_mesh_by_param_dict(self.flame_param)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = (
                torch.tensor(self.cam.world_view_transform).float().cuda().T
            )
            full_proj_transform = (
                torch.tensor(self.cam.full_proj_transform).float().cuda().T
            )
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()

        return Cam
