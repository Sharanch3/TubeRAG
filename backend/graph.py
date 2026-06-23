import sqlite3
from typing import List
from backend.state import RAGState
from backend.vectore_store import get_retriever
from backend.base_model import base_model
from backend.prompt import RAG_PROMPT
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END



retriever = get_retriever()



#NODES
def retrieval_node(state: RAGState) ->dict:

    query: str = state["messages"][-1].content

    docs: List[Document] = retriever.invoke(query)

    context = "\n\n".join(
        f"[{doc.metadata.get('timestamp', 'na')}]\n{doc.page_content}"
        for doc in docs
    )


    return{

        "query": query,

        "retrieved_docs": docs,

        "context": context,

    }




def generate_node(state: RAGState) ->dict:

    chain = RAG_PROMPT | base_model


    response = chain.invoke({
        "context": state.get("context", ""),
        
        "query": state.get("query", "")
    })



    return{

        "messages": [AIMessage(content= response.content)]
    }




#GRAPH
def build_graph():

    graph = StateGraph(state_schema= RAGState)

    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generate_node)
    

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)


    conn = sqlite3.connect(database= "backend/chatbot.db", check_same_thread= False)
    checkpointer = SqliteSaver(conn= conn)
    

    return graph.compile(checkpointer= checkpointer)




if __name__ == "__main__":

    # from pathlib import Path
    
    # ROOT_DIR= Path(__file__).parent.parent.resolve()
    
    # compiled_graph = build_graph()
    # print(f"Graph compiled with nodes: {list(compiled_graph.get_graph().nodes)}")

    # output_path = ROOT_DIR / "architecture/graph.png"
    # output_path.write_bytes(compiled_graph.get_graph().draw_mermaid_png())
    # print(f"Graph diagram saved to {output_path}")

    from langchain_core.messages import HumanMessage

    chatbot = build_graph()

    response = chatbot.invoke(
        {"messages": [HumanMessage(content="What is the capital of India?")]},
        config= {'configurable': {'thread_id': 'thread-1'}}
        
        )['messages'][-1].content
    
    print(response)