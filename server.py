import json
import base64
import numpy as np
import cv2
import asyncio
import requests
import uuid
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from gaussian_renderer import render
from server_utils import LocalViewer, interpolate_blendshapes, Config
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_websockets = set()
# FastAPI App setup remains the same...

def sanitize_filename(text, max_length=20):
    """Sanitize and truncate text to use as a filename."""
    safe_text = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in text)
    return safe_text[:max_length].strip().replace(" ", "_")

def get_session_identifier(token):
    try:
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.post("https://api.lifeguruai.com/api/GetSessionIdentifier", headers=headers)
        response.raise_for_status()
        response_data = response.json()
        identifier = response_data.get("identifier", str(uuid.uuid4()))
        
        return identifier
    except Exception as e:
        print(f"[ERROR] Identifier API Error: {str(e)}")
        return str(uuid.uuid4())

def generate_llm_response(query, identifier, audio_request_id, token):
    try:
        data = {
            "query": query,
            "sessionId": identifier,
            "audio_request_id": audio_request_id
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"  # Add content type header
        }
        
        response = requests.post(
            "https://api.lifeguruai.com/api/ConversationWithTrigger", 
            json=data, 
            headers=headers, 
            stream=True
        )
        response.raise_for_status()
        
        # Process the streaming response
        compiled_response = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                compiled_response += chunk + "\n"
                print(chunk, end="\n", flush=True)  # Stream response in real-time
        
        return compiled_response
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return str(e)
def get_visemes_from_text(text):
    try:
        response = requests.post("https://api.lifeguruai.com/api/TTS_Avatar", 
                               json={"text": text}, 
                               timeout=500)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract both visemes and audio data
        visemes = response_data.get("tts_avatar", {}).get("positions_2d", [])
        audio_base64 = response_data.get("tts_avatar", {}).get("voice_b64", "")
        
        return {
            "visemes": visemes,
            "audio": audio_base64
        }
    except Exception as e:
        print(f"[ERROR] Viseme API Error: {str(e)}")
        return {"visemes": [], "audio": ""}

def convert_visemes_to_flame(visemes):
    try:
        viseme_mapping = {
            "sil": 0,   # Silence (neutral face)

            # Bilabials (Lips close together)
            "p": 21,     # "p" as in "pat"
            "b": 21,     # "b" as in "bat"
            "m": 21,     # "m" as in "mat"

            # Labiodentals (Lips touch teeth)
            "f": 18,     # "f" as in "fish"
            "v": 18,     # "v" as in "van"

            # Dentals/Interdentals (Tongue between teeth)
            "T": 17,     # "th" as in "thin"
            "D": 17,     # "th" as in "this"

            # Alveolars (Tongue near alveolar ridge)
            "t": 19,     # "t" as in "top"
            "d": 19,     # "d" as in "dog"
            "n": 19,     # "n" as in "net"
            "s": 15,     # "s" as in "sip"
            "z": 15,     # "z" as in "zoo"

            # Postalveolar (Tongue slightly back from alveolar ridge)
            "S": 15,     # "sh" as in "she"
            "Z": 15,     # "zh" as in "measure"
            "ch": 6,    # "ch" as in "chip"
            "j": 6,     # "j" as in "judge"

            # Velars (Back of tongue against soft palate)
            "k": 20,     # "k" as in "kick"
            "g": 20,     # "g" as in "go"

            # Glottals (Produced at the glottis)
            "h": 12,    # "h" as in "hat"

            # Approximants (Gliding consonants)
            "r": 6,    # "r" as in "red"
            "l": 11,    # "l" as in "let"
            "w": 7,    # "w" as in "wet"
            "y": 15,    # "y" as in "yes"

            # Front vowels (Lips spread)
            "i": 6,    # "ee" as in "see"
            "e": 6,    # "e" as in "bet"
            "E": 5,    # "æ" as in "cat"
            
            # Central vowels (Neutral position)
            "@": 9,    # Schwa as in "sofa"

            # Back vowels (Lips rounded)
            "o": 13,    # "o" as in "boat"
            "O": 3,    # "aw" as in "law"
            "u": 7,    # "oo" as in "food"

            # Nasals & Stops (Special Cases)
            "ng": 20,   # "ng" as in "sing"
            "x": 21     # "x" as in "Bach" (common in German/Scottish)
        }
        flame_parameters = {
            0:  {"jaw": [0.0, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [0.0, 0.0, 0.0, 0.0, 0.0]},
            1:  {"jaw": [0.18, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [0.55, -0.49, 0.0, 1.80, 0.08]},
            2:  {"jaw": [0.18, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [0.55, -0.49, 0.0, 2.20, 0.08]},
            3:  {"jaw": [0.30, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [-2.0, 0.13, 0.0, -3.0, 0.08]},
            4:  {"jaw": [0.09, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [0.82, -0.90, 0.0, 2.72, 0.08]},
            5:  {"jaw": [0.09, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.38, -0.90, 0.0, 1.80, 0.08]},
            6:  {"jaw": [0.06, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.38, -0.90, 0.0, 1.80, 0.08]},
            7:  {"jaw": [0.09, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [-2.10, 0.25, 0.0, 1.36, 0.08]},
            8:  {"jaw": [0.09, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [-0.26, 0.00, 0.0, 1.10, 0.08]},
            9:  {"jaw": [0.16, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.50, -0.37, 0.0, 1.10, 0.08]},
            10: {"jaw": [0.10, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [-1.68, 0.25, 0.0, 1.92, 0.08]},
            11: {"jaw": [0.16, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.50, -1.05, 0.0, 2.00, 0.08]},
            12: {"jaw": [0.05, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.50, -1.50, 0.0, 2.00, 0.08]},
            13: {"jaw": [0.17, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [-1.50, 0.0, 0.0, 0.0, 0.0]},
            14: {
                "jaw": [0.05, 0.0, 0.0],
                "neck": [0.0, 0.0, 0.0],
                "eyes": [0.0, 0.0, 0.0],
                "expr": [1.50, -0.67, 0.0, 2.00, 0.08]
            },

            15: {
                "jaw": [0.03, 0.0, 0.0],
                "neck": [0.0, 0.0, 0.0],
                "eyes": [0.0, 0.0, 0.0],
                "expr": [1.50, -0.30, 0.0, 2.10, 0.08]
            },

            16: {
                "jaw": [0.02, 0.0, 0.0],
                "neck": [0.0, 0.0, 0.0],
                "eyes": [0.0, 0.0, 0.0],
                "expr": [-0.86, 1.20, 0.0, 2.52, 0.08]
            },

            17: {
                "jaw": [0.05, 0.0, 0.0],
                "neck": [0.0, 0.0, 0.0],
                "eyes": [0.0, 0.0, 0.0],
                "expr": [1.91, 0.31, 0.0, 3.00, 0.08]
            },

            18: {
                "jaw": [0.07, 0.0, 0.0],
                "neck": [0.0, 0.0, 0.0],
                "eyes": [0.0, 0.0, 0.0],
                "expr": [1.32, 1.20, 0.0, 0.03, 0.08]
            },
            19: {"jaw": [0.07, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.32, -0.71, 0.0, 3.00, 0.08]},
            20: {"jaw": [0.19, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [1.05, -1.15, 0.0, 1.80, 0.00]},
            21: {"jaw": [0.0, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "eyes": [0.0, 0.0, 0.0], "expr": [0.70, 0.0, 0.0, 0.0, 0.0]}
        }

        result = []
        result.append({"time": 0, "parameters": flame_parameters[0]})
        for v in visemes:
            viseme_value = viseme_mapping.get(v["value"], 0)
            if viseme_value in flame_parameters:
                result.append({"time": v["time"], "parameters": flame_parameters[viseme_value]})
            else:
                print(f"Skipping for: {viseme_value}")
        results = interpolate_blendshapes(result)
        return results
    except Exception as e:
        print("Error in conversion: ", str(e))
        return visemes


class LipsyncVideoGenerator:
    def __init__(self):
        self.cfg = Config(
            point_path=Path("media/306/point_cloud.ply"),
            save_folder=Path("output_frames"),
            fps=30,
            demo_mode=True
        )
        self.viewer = LocalViewer(self.cfg)
        self.visemes = []
        self.frame_idx = 0
        self.is_streaming = False
        self.audio_data = None
        self.total_duration = 0
        self.cam = None  # Will be initialized when needed
        
        # Initialize background color tensor
        self.background_color = torch.tensor(self.cfg.background_color).cuda()
        
        # Frame cache with limited size
        self.frame_cache = {}
        self.max_cache_size = 100  # Reduced cache size for better memory management

    def set_visemes_and_audio(self, visemes, audio_base64):
        """Set visemes and audio data"""
        try:
            # Clear previous cache and reset states
            self.frame_cache.clear()
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            interpolated_visemes = convert_visemes_to_flame(visemes)
            self.visemes = interpolated_visemes
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
            print(f"Error in set_visemes_and_audio: {str(e)}")
            self.is_streaming = False

    def _generate_single_frame(self, frame_idx):
        """Generate a single frame with proper CUDA synchronization"""
        if frame_idx >= len(self.visemes):
            return None

        try:
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            current_frame = self.visemes[frame_idx]
            self.viewer.apply_blendshapes(current_frame)

            # Ensure we're using the correct device
            with torch.cuda.device(self.background_color.device):
                render_output = render(
                    self.cam,
                    self.viewer.gaussians,
                    self.cfg.pipeline,
                    self.background_color
                )

                frame_tensor = render_output.get("render")
                if frame_tensor is None:
                    return None

                # Process tensor with proper synchronization
                with torch.no_grad():
                    frame_tensor = frame_tensor.detach()
                    frame_image = (torch.clamp(frame_tensor, 0, 1) * 255)
                    frame_image = frame_image.to(torch.uint8)
                    frame_image = frame_image.permute(1, 2, 0).cpu()
                    
                # Synchronize before numpy conversion
                torch.cuda.synchronize()
                frame_image = frame_image.numpy()

            # Convert to base64
            img = Image.fromarray(frame_image)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            timestamp = current_frame["time"] / 1000.0
            is_last_frame = frame_idx >= len(self.visemes) - 1

            return {
                "frame": frame_base64,
                "timestamp": timestamp,
                "is_last": is_last_frame
            }

        except Exception as e:
            print(f"Frame generation error: {str(e)}")
            torch.cuda.empty_cache()  # Clear CUDA cache on error
            return None

    async def get_next_frame(self):
        """Get next frame with error handling"""
        if not self.visemes or self.frame_idx >= len(self.visemes):
            return None

        try:
            # Check cache first
            if self.frame_idx in self.frame_cache:
                frame_data = self.frame_cache[self.frame_idx]
                del self.frame_cache[self.frame_idx]  # Free memory
                self.frame_idx += 1
                return frame_data

            # Generate new frame
            frame_data = self._generate_single_frame(self.frame_idx)
            self.frame_idx += 1
            
            # Cache next frame if possible
            if self.frame_idx < len(self.visemes) and len(self.frame_cache) < self.max_cache_size:
                next_frame = self._generate_single_frame(self.frame_idx)
                if next_frame:
                    self.frame_cache[self.frame_idx] = next_frame

            return frame_data

        except Exception as e:
            print(f"Error in get_next_frame: {str(e)}")
            return None

    async def stream_video(self, websocket: WebSocket):
        """Stream video with improved error handling"""
        try:
            # Send audio data
            await websocket.send_json({
                "type": "audio",
                "data": self.audio_data,
                "duration": self.total_duration
            })

            # Wait for client confirmation
            await websocket.receive_text()

            # Stream frames
            while self.is_streaming:
                frame_data = await self.get_next_frame()
                if frame_data is None:
                    self.is_streaming = False
                    break

                await websocket.send_json({
                    "type": "frame",
                    "data": frame_data["frame"],
                    "timestamp": frame_data["timestamp"],
                    "is_last": frame_data["is_last"]
                })

                if frame_data["is_last"]:
                    self.is_streaming = False
                    break

        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            print(f"Streaming error: {str(e)}")
        finally:
            self.is_streaming = False
            torch.cuda.empty_cache()  # Ensure CUDA memory is cleared

    def cleanup(self):
        """Clean up resources"""
        self.frame_cache.clear()
        torch.cuda.empty_cache()
        self.is_streaming = False
        self.frame_idx = 0
# class LipsyncVideoGenerator:
#     def __init__(self):
#         self.cfg = Config(
#             point_path=Path("media/306/point_cloud.ply"),
#             save_folder=Path("output_frames"),
#             fps=30,
#             demo_mode=True
#         )
#         self.viewer = LocalViewer(self.cfg)
#         self.visemes = []
#         self.frame_idx = 0
#         self.is_streaming = False
#         self.start_time = None
#         self.audio_data = None
#         self.total_duration = 0
#         self.cam = None

#     def set_visemes_and_audio(self, visemes, audio_base64):
#         """Set both visemes and audio data"""
#         interpolated_visemes = convert_visemes_to_flame(visemes)
#         self.visemes = interpolated_visemes
#         self.audio_data = audio_base64
#         self.frame_idx = 0
#         self.is_streaming = True
#         self.start_time = None
        
#         # Calculate total duration from visemes in seconds
#         if self.visemes:
#             self.total_duration = max(v["time"] for v in self.visemes) / 1000.0
#             print(f"Total duration: {self.total_duration} seconds")

#     async def get_next_frame(self):
#         """Generate a single frame with preserved colors."""
#         if not self.visemes or self.frame_idx >= len(self.visemes):
#             return None

#         try:
#             current_frame = self.visemes[self.frame_idx]
#             self.viewer.apply_blendshapes(current_frame)

#             # cam = self.viewer.prepare_camera()
#             render_output = render(
#                 self.cam,
#                 self.viewer.gaussians,
#                 self.cfg.pipeline,
#                 torch.tensor(self.cfg.background_color).cuda()
#             )

#             frame_tensor = render_output.get("render")
#             if frame_tensor is None:
#                 return None

#             # Convert tensor to image without changing color values
#             frame_image = (np.clip(render_output["render"].detach().permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

#             img = Image.fromarray(frame_image)

#             # Save to a BytesIO buffer (instead of disk)
#             buffer = BytesIO()
#             img.save(buffer, format="PNG")  # PNG ensures lossless color preservation
#             buffer.seek(0)

#             # Convert to base64
#             frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

#             # Get timestamp in seconds
#             timestamp = current_frame["time"] / 1000.0

#             # Check if this is the last frame
#             is_last_frame = self.frame_idx >= len(self.visemes) - 1

#             self.frame_idx += 1

#             return {
#                 "frame": frame_base64,
#                 "timestamp": timestamp,
#                 "is_last": is_last_frame
#             }


#         except Exception as e:
#             print(f"Frame generation error: {str(e)}")
#             return None
#     async def stream_video(self, websocket: WebSocket):
#         """Stream synchronized video frames and audio"""
#         try:
#             # Send audio first
#             await websocket.send_json({
#                 "type": "audio",
#                 "data": self.audio_data,
#                 "duration": self.total_duration
#             })
#             # print("Fps: ", self.cfg.fps)

#             # Wait for client to confirm audio loaded
#             await websocket.receive_text()
#             if not self.cam:
#                 self.cam = self.viewer.prepare_camera()
#             # Stream frames
#             while self.is_streaming:
#                 frame_data = await self.get_next_frame()
#                 if frame_data is None:
#                     self.is_streaming = False
#                     break

#                 await websocket.send_json({
#                     "type": "frame",
#                     "data": frame_data["frame"],
#                     "timestamp": frame_data["timestamp"],
#                     "is_last": frame_data["is_last"]
#                 })

#                 if frame_data["is_last"]:
#                     self.is_streaming = False
#                     break

#                 # await asyncio.sleep(1 / self.cfg.fps)
#                 # print("Frame: ", self.frame_idx)
                
#         except WebSocketDisconnect:
#             print("WebSocket disconnected")
#         except Exception as e:
#             print(f"Streaming error: {str(e)}")
#         finally:
#             self.is_streaming = False

# Create video generator instance
video_generator = LipsyncVideoGenerator()

@app.get("/")
async def health_check():
    return {"message": "WebSocket Lipsync Server Running!"}

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    identifier = get_session_identifier(token=token)
    if not identifier:
        await websocket.send_json({
            "type": "authentication",
            "message": "Failed to authenticate the connection"
        })
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            print("Request data: ", data)
            request_data = data.get("data")
            text = request_data.get("text")
            audio_request_id = request_data.get("id", None)
            
            if text:
                print(f"[INFO] Received text: {text}")
                
                llm_response = generate_llm_response(text, identifier, audio_request_id, token)
                # Get visemes and audio from API
                response_data = get_visemes_from_text(llm_response)
                visemes = response_data["visemes"]
                audio_base64 = response_data["audio"]
                
                if visemes and audio_base64:
                    # Set both visemes and audio
                    video_generator.set_visemes_and_audio(visemes, audio_base64)
                    await video_generator.stream_video(websocket)
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to get visemes or audio data"
                    })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to input data"
                })
            
    except Exception as e:
        print(f"[ERROR] WebSocket Error: {e}")
    finally:
        video_generator.is_streaming = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)