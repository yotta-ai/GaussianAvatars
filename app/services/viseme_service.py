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
        """
        Interpolates blendshapes based on timestamps to match the specified FPS.

        The function expects each element of `blendshapes` to have a 'time' key and
        a 'parameters' key containing a dict of numeric lists. It does not hardcode
        any parameter names. In case of duplicate times, the parameters are averaged.
        """
        try:
            if not blendshapes:
                return []

            # --- STEP 1: Sort the blendshapes by time ---
            blendshapes = sorted(blendshapes, key=lambda b: b["time"])

            # --- STEP 2: Group and average duplicate timestamps ---
            grouped = {}  # key: time, value: list of parameters dicts
            for b in blendshapes:
                t = b["time"]
                grouped.setdefault(t, []).append(b["parameters"])

            unique_times = sorted(grouped.keys())
            averaged_blendshapes = []
            for t in unique_times:
                param_list = grouped[t]
                # Determine dynamic keys (assuming all dicts for a given time share the same keys)
                keys = param_list[0].keys()
                averaged_params = {}
                for key in keys:
                    # Convert each parameter's value to a numpy array and average along the 0-axis.
                    arr = np.array([params[key] for params in param_list], dtype=float)
                    avg_val = np.mean(arr, axis=0)
                    # Convert the result to a list if it is an array.
                    averaged_params[key] = (
                        avg_val.tolist() if isinstance(avg_val, np.ndarray) else avg_val
                    )
                averaged_blendshapes.append({"time": t, "parameters": averaged_params})

            # Replace the original blendshapes with the averaged version.
            blendshapes = averaged_blendshapes

            # --- STEP 3: Setup interpolation ---
            # Extract sorted timestamps.
            timestamps = [b["time"] for b in blendshapes]
            start, end = timestamps[0], timestamps[-1]
            step = 1000 / fps  # converting fps to a step in milliseconds

            # Create frame times ensuring that the final time (end) is included.
            frame_times = list(np.arange(start, end, step))
            if not np.isclose(frame_times[-1], end):
                frame_times.append(end)

            # Dynamically extract parameter keys.
            keys = blendshapes[0]["parameters"].keys()

            # Create an interpolation function for each parameter key.
            interpolators = {}
            for k in keys:
                # Gather the values for key `k` over all blendshapes.
                values = np.array(
                    [b["parameters"][k] for b in blendshapes], dtype=float
                )
                # Build the interpolation function.
                interp_func = interp1d(
                    timestamps, values, axis=0, kind="linear", fill_value="extrapolate"
                )
                interpolators[k] = interp_func

            # --- STEP 4: Interpolate for each frame time ---
            interpolated_blendshapes = []
            for t in frame_times:
                frame_params = {}
                for k in keys:
                    # Evaluate and, if needed, convert numpy arrays to lists.
                    val = interpolators[k](t)
                    frame_params[k] = (
                        val.tolist() if isinstance(val, np.ndarray) else val
                    )
                # Note: The returned structure is similar to the input:
                # a dict with 'time' and 'parameters'
                interpolated_blendshapes.append(
                    {"time": int(round(t)), "parameters": frame_params}
                )

            return interpolated_blendshapes

        except Exception as e:
            logging.error(f"Error in interpolation: {e}")
            return blendshapes
