import streamlit as st
from frontend.utils import color_timestamps
from frontend.sidebar import render_sidebar
from frontend.page_config import page_config
from langchain_core.messages import HumanMessage
from frontend.session_state import render_session_state
from frontend.chat_history import render_chat_history



def main():

    page_config()

    render_session_state()

    st.title("🤖 Youtube Assistant")

    render_sidebar()

    render_chat_history()


    if not st.session_state["video_loaded"]:

        with st.container(border=True):
            st.markdown(
                """
                **`📄` No Transcript Loaded**

                >Upload **YouTube URL** in the **sidebar** and start asking questions
                """
            )

        st.stop() 
    
    
    user_input = st.chat_input(placeholder="Ask me anything about the video...")
    
    if user_input:

        CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

        #Display User Input
        with st.chat_message("user"):
            st.write(user_input)

        #Persist User Input
        st.session_state["chat_history"].append({'role': 'user', 'content': user_input})


       
        result = st.session_state["graph"].invoke(
                {"messages": [HumanMessage(content= user_input)]},
                config = CONFIG
            )

        response = result["messages"][-1].content
        
        #Display AI Response
        with st.chat_message("assistant"):
            st.markdown(
                color_timestamps(text= response),
                unsafe_allow_html= True
            )


        #Persist AI response
        st.session_state["chat_history"].append({'role':'assistant', 'content': response})





if __name__ == "__main__":

    main()

