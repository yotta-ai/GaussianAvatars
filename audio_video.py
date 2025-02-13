from moviepy import VideoFileClip, AudioFileClip

# File paths
video_path = "audio2.mp4"  # Replace with your video file
audio_path = "audio2.mp3"  # Replace with your audio file
output_path = "audio_audio2.mp4"  # Output file

# Load video and audio
video = VideoFileClip(video_path)
audio = AudioFileClip(audio_path)

# Set the audio duration to match the video duration
audio = audio.with_duration(video.duration)

# Combine video with audio
final_video = video.with_audio(audio)

# Save the output video
final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

print("Video with audio saved successfully!")
