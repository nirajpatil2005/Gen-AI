import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
import re
from dotenv import load_dotenv

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="YouTube Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS for styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    h1 { color: #0d6efd; }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 8px;
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
        padding: 8px;
    }
    .success-message { 
        color: #198754; 
        font-weight: bold;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 8px;
    }
    .error-message { 
        color: #dc3545; 
        font-weight: bold;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 8px;
    }
    .warning-message {
        color: #ff8f00;
        font-weight: bold;
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 8px;
    }
    .info-message {
        color: #1976d2;
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🎥 YouTube Video Assistant")
st.markdown("Ask questions about any YouTube video content")

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([^?]+)',
        r'([a-zA-Z0-9_-]{11})'  # Standalone video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Configure your video analysis")
    
    youtube_url = st.text_input(
        "🔗 YouTube Video URL",
        placeholder="Paste YouTube link here...",
        help="Supports full URLs, short URLs, and video IDs"
    )
    
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your key from console.groq.com"
    )
    
    model_name = st.selectbox(
        "🤖 AI Model",
        ["mixtral-8x7b-32768", "llama2-70b-4096"],  # Removed gemma-7b-it to simplify
        index=0,
        help="Select the AI model for analysis"
    )
    
    if st.button("🚀 Process Video", use_container_width=True):
        st.session_state.process_clicked = True

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if 'process_clicked' in st.session_state and st.session_state.process_clicked:
        if not youtube_url:
            st.error("Please enter a YouTube URL", icon="❌")
        elif not groq_api_key:
            st.error("Please enter your Groq API key", icon="❌")
        else:
            try:
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL format", icon="❌")
                else:
                    with st.spinner("🔍 Analyzing video content..."):
                        try:
                            # Get transcript using the correct method
                            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                            texts = [t['text'] for t in transcript]
                            
                            # Split text
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1500,
                                chunk_overlap=300,
                                separators=["\n\n", "\n", ". ", "? ", "! "]
                            )
                            chunks = splitter.create_documents(texts)
                            
                            # Create embeddings using the updated import
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                            vector_store = FAISS.from_documents(chunks, embedding=embeddings)
                            
                            # Create retriever
                            retriever = vector_store.as_retriever(
                                search_type="mmr",
                                search_kwargs={'k': 6, 'lambda_mult': 0.5}
                            )
                            
                            # Initialize LLM
                            llm = ChatGroq(
                                groq_api_key=groq_api_key,
                                model_name=model_name,
                                temperature=0.1
                            )
                            
                            # Create prompt template
                            prompt = PromptTemplate(
                                template="""
                                You are a YouTube video expert analyzing this content:
                                {context}
                                
                                Question: {question}
                                
                                Provide a helpful answer following these rules:
                                1. Be accurate and concise (1-3 sentences)
                                2. Include timestamps when relevant
                                3. If unsure, say "Not mentioned in the video"
                                4. Maintain a friendly, professional tone
                                """,
                                input_variables=['context', 'question']
                            )
                            
                            # Create chain
                            def format_docs(docs):
                                return "\n\n".join(doc.page_content for doc in docs)
                            
                            chain = (
                                RunnableParallel({
                                    "context": retriever | format_docs,
                                    "question": RunnablePassthrough()
                                })
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                            
                            st.session_state['chain'] = chain
                            st.session_state['video_processed'] = True
                            st.session_state['video_id'] = video_id
                            st.success(f"✅ Successfully processed video: {video_id}", icon="✅")
                            
                        except TranscriptsDisabled:
                            st.error("Transcripts are disabled for this video", icon="🔇")
                        except Exception as e:
                            if "No transcripts were found" in str(e):
                                st.error("No English transcript available", icon="🌐")
                            else:
                                st.error(f"Error processing video: {str(e)}", icon="⚠️")

            except Exception as e:
                st.error(f"Error: {str(e)}", icon="⚠️")

with col2:
    st.markdown("### 💡 How to use")
    st.markdown("""
    1. Paste YouTube URL
    2. Enter Groq API key
    3. Select AI model
    4. Click Process Video
    5. Ask questions below
    """)
    
    st.markdown("### 🌟 Tips")
    st.markdown("""
    - Works best with English videos
    - For longer videos, processing may take a moment
    - Try questions like:
      - "Summarize the key points"
      - "What was said about [topic]?"
      - "Find mentions of [concept]"
    """)

# Chat interface
if st.session_state.get('video_processed', False):
    st.divider()
    st.subheader(f"💬 Ask About Video: {st.session_state['video_id']}")
    
    user_question = st.text_area(
        "Type your question here...",
        height=100,
        placeholder="What would you like to know about this video?",
        key="question_input"
    )
    
    if st.button("🔎 Get Answer", use_container_width=True):
        if user_question:
            with st.spinner("🧠 Generating answer..."):
                try:
                    response = st.session_state['chain'].invoke(user_question)
                    st.markdown("### 📝 Answer")
                    st.info(response)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}", icon="⚠️")
        else:
            st.warning("Please enter a question", icon="⚠️")