from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document




class RAGState(TypedDict):

    messages: Annotated[List[BaseMessage], add_messages]

    query: str
    
    retrieved_docs: List[Document] | None

    context: str

    answer: str

