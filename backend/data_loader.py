from typing import List
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader





def format_seconds_to_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    
    return f"{minutes}:{secs:02d}"



def load_transcript(youtube_url: str, chunk_size: int = 25) ->List[Document]:
    
    video_id = YoutubeLoader.extract_video_id(youtube_url=youtube_url)

    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.list(video_id=video_id)

    transcript_en = transcript_list.find_transcript(language_codes=["en"])
    raw_transcript = transcript_en.fetch()

    documents = []

    for i in range(0, len(raw_transcript), chunk_size):
        
        chunk = raw_transcript[i:i + chunk_size]

        text = " ".join(
            snippet.text
            for snippet in chunk
        )

        start_time = chunk[0].start
        end_time = chunk[-1].start + chunk[-1].duration 

        doc = Document(
            page_content=text,  
            metadata={
                "timestamp": f"[{format_seconds_to_time(start_time)} - {format_seconds_to_time(end_time)}]"
            }
        )

        documents.append(doc)

    return documents