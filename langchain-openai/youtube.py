from youtube_transcript_api import YouTubeTranscriptApi

# Function to format time (seconds → hh:mm:ss)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to get only English transcript and align into 30s segments
def get_english_transcript(video_id, filename="transcript.txt", segment_duration=30):
    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Fetch only English transcript
        transcript = transcript_list.find_transcript(['en']).fetch()
        
        # Organize transcript into 30-second segments
        segments = []
        current_segment = []
        start_time = 0

        for entry in transcript:
            if entry["start"] - start_time >= segment_duration:
                # Save the previous segment as a single line
                segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))
                # Start a new segment
                start_time = entry["start"]
                current_segment = []
            
            current_segment.append(entry["text"])

        # Save the last segment
        if current_segment:
            segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))

        # Save to a text file
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(segments))

        print(f"English transcript saved to {filename}")
    except Exception as e:
        print(f"Error: {e}")

# Example: Extract English transcript from a YouTube video
video_url = "https://www.youtube.com/watch?v=8O5kX73OkIY"  # Replace with your video URL
video_id = video_url.split("v=")[-1].split("&")[0]  # Handles additional parameters
get_english_transcript(video_id)
