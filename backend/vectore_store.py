import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()



ROOT_DIR: Final[Path] = Path(__file__).parent.parent.resolve()



_base_embedding = OpenAIEmbeddings(
    model= "text-embedding-3-small",
    api_key= os.getenv("OPENAI_API_KEY")
)


vs = Chroma(
    collection_name= "transcripts",
    embedding_function= _base_embedding,
    persist_directory= (ROOT_DIR / "vector-store").as_posix(),

)



def get_retriever():

    return vs.as_retriever(
                search_type = "mmr",
                search_kwargs = {'k': 4, 'fetch_k': 20, 'lambda_mult': 0.3}
            )
   