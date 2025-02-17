import json
import requests
import logging
from typing import Dict, List, Optional
from app.core.config import settings, config
from app.core.constants import VISEME_MAPPING, FLAME_PARAMETERS
from scipy.interpolate import interp1d  # Import scipy
import numpy as np


class VisemeService:
    def __init__(self):
        self.tts_avatar_api_url = settings.TTS_AVATAR_API_URL

    def get_visemes_and_audio_from_text(self, text: str) -> Dict:
        """Gets visemes and audio data from the text-to-speech API."""
        try:
            response = requests.post(
                self.tts_avatar_api_url,
                json={"text": text},
                timeout=60,  # Increased timeout, can adjust
            )
            response.raise_for_status()
            response_data = response.json()
            return {
                "visemes": response_data.get("tts_avatar", {}).get("positions_2d", []),
                "audio": response_data.get("tts_avatar", {}).get("voice_b64", ""),
            }
        except requests.RequestException as e:
            logging.error(f"[ERROR] Viseme API Error: {e}")
            return {"visemes": [], "audio": ""}

    def convert_visemes_to_flame_parameters(self, visemes: List[Dict]) -> List[Dict]:
        """Converts visemes to FLAME parameters, including interpolation."""
        result = []
        result.append({"time": 0, "parameters": FLAME_PARAMETERS[0]})

        for viseme in visemes:
            viseme_value = VISEME_MAPPING.get(viseme["value"], 0)  # Default to silence
            if viseme_value in FLAME_PARAMETERS:
                result.append(
                    {
                        "time": viseme["time"],
                        "parameters": FLAME_PARAMETERS[viseme_value],
                    }
                )
            else:
                logging.warning(f"Skipping unknown viseme value: {viseme_value}")
        # save visemes and result in json file
        with open("visemes.json", "w") as f:
            json.dump(visemes, f)
        with open("flame_params.json", "w") as f:
            json.dump(result, f)

        if len(result) > 1:
            result = self.interpolate_blendshapes(result)  # Use interpolation
        return result

    def interpolate_blendshapes(
        self, blendshapes: List[Dict], fps: int = config.fps
    ) -> List[Dict]:
        """Interpolates blendshapes based on timestamps to match FPS."""
        try:
            if not blendshapes:  # Handle empty input list
                return []

            timestamps = [b["time"] for b in blendshapes]
            total_duration = timestamps[-1]
            frame_times = np.arange(timestamps[0], total_duration, 1000 / fps)

            interpolated_blendshapes = []
            keys = blendshapes[0]["parameters"].keys()

            # Create interpolation functions for each blendshape key
            interpolators = {k: [] for k in keys}
            for k in keys:
                values = np.array([b["parameters"][k] for b in blendshapes])
                interp_func = interp1d(
                    timestamps, values, axis=0, kind="linear", fill_value="extrapolate"
                )
                interpolators[k] = [interp_func(t) for t in frame_times]

            # Convert interpolated values into frame-wise blendshapes
            for i, frame_time in enumerate(frame_times):
                frame_blendshape = {k: interpolators[k][i].tolist() for k in keys}
                frame_blendshape["time"] = int(frame_time)
                interpolated_blendshapes.append(frame_blendshape)

            return interpolated_blendshapes
        except Exception as e:
            logging.error(f"Error in interpolation: {e}")
            return blendshapes
