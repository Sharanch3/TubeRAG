import uuid
import streamlit as st
from backend.data_loader import load_transcript
from backend.vectore_store import vs
from backend.graph import build_graph




def render_sidebar() ->None:

    with st.sidebar:

        st.markdown("## `▶` YouTube Q&A")
        st.divider()


        input_url = st.text_input(
            label= "Youtube URL",
            placeholder = " ",

        )

        load_transcript_btn = st.button(label="Load Transcript", type= "primary", width= "content")


        if load_transcript_btn and input_url.strip():

            try:
            
                with st.spinner("Fetching and Loading Transcript..."):

                    chunks = load_transcript(youtube_url= input_url)

                    vs.reset_collection()
                    vs.add_documents(documents= chunks)

                    st.session_state["video_loaded"] = True
                    st.session_state["current_url"] = input_url
                    st.session_state["chat_history"] = []
                    st.session_state["thread_id"] = str(uuid.uuid4())
                    st.session_state["graph"] = build_graph()

                    
                    st.toast(body="Successfully Loaded!", icon= "✅", duration= "short")
            

            except Exception as e:

                st.error(str(e))

                
        if st.session_state["video_loaded"]:
                
            st.divider()

            st.markdown(body= "## `▶` Current Video")
            st.markdown(body=f"* {st.session_state['current_url']}")
                

            if st.button("Load New Video"):
                    
                st.session_state["video_loaded"] = False
                st.session_state["current_url"] = "" 
                st.session_state["chat_history"] = []
                st.session_state["thread_id"] = str(uuid.uuid4())
            
                st.rerun()

                





        