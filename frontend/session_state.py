import uuid
import streamlit as st



def render_session_state():

    if "graph" not in st.session_state:
        st.session_state["graph"] = None

    
    if "video_loaded" not in st.session_state:
        st.session_state["video_loaded"] = False

    
    if "current_url" not in st.session_state:
        st.session_state["current_url"] = ""
    
    
    if "thread_id" not in st.session_state:
        
        thread_id = str(uuid.uuid4())

        st.session_state["thread_id"] = thread_id


    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


    



    
    