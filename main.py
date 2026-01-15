import streamlit as st
from rag_utils import process_transcript, generate_answer


# Page configuration
st.set_page_config(
    page_title="TubeRAG",
    page_icon="📹",
    layout="centered"
)

# Title
st.title("📹 TUBERAG")
st.markdown("### Ask questions about any YouTube video")
st.divider()

placeholder = st.empty()

# Sidebar for URL input
with st.sidebar:
    url = st.text_input(
        label="YouTube URL",
        help="Enter a valid YouTube video URL"
    )
    process_button = st.button("Process Video", type="primary", use_container_width=True)


# Process video when button is clicked
if process_button:
    if not url.strip():
        st.sidebar.error("⚠️ Please provide a valid URL")
    else:
        for status in process_transcript(url= url):
            placeholder.write(status)


# Show instructions
if not process_button:
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### How to use:
    1. Paste a YouTube video URL in the sidebar
    2. Click 'Process Video' to extract and analyze the transcript
    3. Ask any question about the video content
    4. Get AI-powered answers based on the video transcript
    """)
    
    

# Question-answering section
st.subheader("💬 Ask a Question")

query = st.text_area(
    label="Your Question",
    placeholder="Ask anything about the video...",
    help="Enter your question about the video content",
    height=100,
    label_visibility="collapsed"
)

query_button = st.button("Get Answer", type="primary")

if query_button:
    if not query.strip():
        st.error("⚠️ Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            try:
                response = generate_answer(query=query)

                st.subheader("💡 Answer:")
                st.success(response)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")