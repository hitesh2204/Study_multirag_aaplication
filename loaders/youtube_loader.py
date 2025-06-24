

# loaders/youtube_loader.py
from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(video_url_or_id: str) -> str:
    # Extract just the video ID
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        video_id = video_url_or_id.rstrip("/").split("/")[-1].split("?")[0]
    else:
        video_id = video_url_or_id
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(item["text"] for item in transcript)
