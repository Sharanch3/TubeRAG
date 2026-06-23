import streamlit as st
from utils import color_timestamps

def render_chat_history():

    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):
            
            st.markdown(color_timestamps(message["content"]), unsafe_allow_html= True)