import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables import(
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
) 

load_dotenv()


# CONFIGURATION CONSTANTS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "youtube-transcript"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTORDB_DIR = Path(__file__).parent / "resources/vector_store"


# INITIALIZE OBJECTS
llm = None
vector_store = None


def initialize_components() -> None:
    """Initialize LLM and vector store components."""
    
    global llm, vector_store

    if llm is None:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )

    if vector_store is None:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )

        vector_store = Chroma(
            collection_name="transcripts",
            embedding_function=embeddings,
            persist_directory=str(VECTORDB_DIR) 
        )


def fetch_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    
    try:
        return YoutubeLoader.extract_video_id(url)
    
    except Exception as e:
        raise ValueError(f"Invalid YouTube URL: {str(e)}")


def process_transcript(url: str) -> None:
    """Process YouTube transcript and store in vector database."""
    
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Extracting Video ID..."
    video_id = fetch_video_id(url)

    yield "Extracting Video Transcripts..."

    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.list(video_id= video_id)

        transcript_en = transcript_list.find_transcript(language_codes=["en"])
        
        # Try to get English transcript
        transcript_en = transcript_list.find_transcript(['en'])
        raw_transcript = transcript_en.fetch()

        transcript = ' '.join([doc.text for doc in raw_transcript])

    except TranscriptsDisabled:
        raise RuntimeError("No captions available for this video.")
    
    except NoTranscriptFound:
        raise RuntimeError("English transcript not available.")
    
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {str(e)}")

    yield "Splitting Transcripts into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Create a list of Document objects from a list of texts
    chunks = splitter.create_documents([transcript])

    yield "Embedding and storing chunks..."
    vector_store.add_documents(chunks)

    yield "All setup complete. ✅"


def generate_answer(query: str) -> str:
    """Generate answer to query using RAG pipeline."""

    if vector_store is None:
        raise RuntimeError(
            "Vector database is not initialized. "
            "Please process a YouTube URL first."
        )
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 7, "lambda_mult": 0.5}
    )
    
    # Load Prompt Template
    prompt = load_prompt("./prompt_template.json")
    parser = StrOutputParser()

    # CHAIN
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | llm | parser

    response = final_chain.invoke(query)
    
    return response




if __name__ == "__main__":

    url = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    query = "is the topic of aliens discussed in this video? If yes then what was discussed?"

    # Process the transcript
    process_transcript(url=url)

    # Generate and display the answer
    print("\n" + "="*60)
    print("QUERY:", query)
    print("="*60 + "\n")
    
    answer = generate_answer(query=query)
    
    print("ANSWER:")
    print(answer)
    print("\n" + "="*60)
    
        
