import os
import json
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# Paths
PDF_STORAGE_PATH = 'document_store/pdfs/'
PDF_LIST_PATH = os.path.join(PDF_STORAGE_PATH, "pdf_list.json")

# Ensure directories exist
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Function to format time (seconds → hh:mm:ss)
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to save transcript
def save_transcript(video_name, transcript_text):
    file_path = os.path.join(PDF_STORAGE_PATH, f"{video_name}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(transcript_text)
    return file_path

# Load PDF list
def load_pdf_list():
    if os.path.exists(PDF_LIST_PATH):
        with open(PDF_LIST_PATH, "r") as f:
            return json.load(f)
    return []

# Save PDF list
def save_pdf_list(pdf_list):
    with open(PDF_LIST_PATH, "w") as f:
        json.dump(pdf_list, f)

# Function to extract YouTube Video ID from URL
def extract_video_id(url):
    parsed_url = urlparse(url)
    video_id = None
    
    if parsed_url.netloc in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.netloc in ["youtu.be"]:  # Shortened URL format
        video_id = parsed_url.path.lstrip("/")

    return video_id

# Function to get only English transcript and align into 30s segments
def get_english_transcript(video_id, video_name, segment_duration=30):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en']).fetch()
        
        # Organize transcript into 30-second segments
        segments = []
        current_segment = []
        start_time = 0

        for entry in transcript:
            if entry["start"] - start_time >= segment_duration:
                segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))
                start_time = entry["start"]
                current_segment = []
            
            current_segment.append(entry["text"])

        # Save the last segment
        if current_segment:
            segments.append(f"[{format_time(start_time)}] " + " ".join(current_segment))

        # Save to a text file
        file_path = save_transcript(video_name, "\n".join(segments))
        return file_path, "\n".join(segments)

    except Exception as e:
        return None, f"Error: {e}"

# Streamlit UI
st.title("📹 YouTube Transcript Extractor")
st.write("Paste a YouTube link to extract and save the transcript.")

# Input field for YouTube link
video_url = st.text_input("🔗 Paste YouTube link here:")

if st.button("🎬 Get Transcript"):
    if video_url:
        try:
            # Extract Video ID
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please check and try again.")
                st.stop()

            # Generate video name
            video_name = f"transcript_{video_id}"

            # Extract transcript
            file_path, transcript_text = get_english_transcript(video_id, video_name)

            if file_path:
                # Update PDF list
                pdf_list = load_pdf_list()
                if video_name not in pdf_list:
                    pdf_list.append(video_name)
                    save_pdf_list(pdf_list)

                st.success(f"✅ Transcript saved as: {video_name}.txt")
                st.download_button(label="📥 Download Transcript", data=transcript_text, file_name=f"{video_name}.txt", mime="text/plain")
            else:
                st.error(transcript_text)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    else:
        st.warning("⚠️ Please enter a valid YouTube link.")
