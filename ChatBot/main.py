import streamlit as st
import os
from gtts import gTTS
from io import BytesIO
from groq import Groq
from datetime import datetime
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr

# Load environment variables
groq_api_key = st.secrets["GROQ_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="Groq Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

def get_custom_prompt():
    """Get custom prompt template based on selected persona"""
    persona = st.session_state.get('selected_persona', 'Default')
    
    personas = {
        'Default': """You are a helpful AI assistant.
                     Current conversation:
                     {history}
                     Human: {input}
                     AI:""",
        'Expert': """You are an expert consultant with deep knowledge across multiple fields.
                    Please provide detailed, technical responses when appropriate.
                    Current conversation:
                    {history}
                    Human: {input}
                    Expert:""",

        'Creative': """You are a creative and imaginative AI that thinks outside the box.
                      Feel free to use metaphors and analogies in your responses.
                      Current conversation:
                      {history}
                      Human: {input}
                      Creative AI:"""
    }
    
    return PromptTemplate(
        input_variables=["history", "input"],
        template=personas[persona]
    )

def display_chat_statistics():
    """Display chat statistics in the sidebar."""
    if st.session_state.start_time:
        duration = datetime.now() - st.session_state.start_time
        st.subheader("📊 Chat Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.chat_history))
        with col2:
            st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")

def get_voice_input():
    """Capture and process voice input"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("⏳ Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"⚠️ Could not request results from the speech recognition service: {e}")
        return ""

def text_to_audio(text):
    """Convert text to audio and return a streamable object"""
    tts = gTTS(text, lang='en')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

def main():
    initialize_session_state()

    # Sidebar Configuration
    with st.sidebar:
        st.title("🛠️ Chat Settings")
        
        # Model selection
        st.subheader("Model Selection")
        model = st.selectbox(
            'Choose your model:',
            ['gemma2-9b-it', 'mixtral-8x7b-32768', 'llama2-70b-4096'],
            help="Select the AI model for your conversation"
        )
        
        # Memory configuration
        st.subheader("Memory Settings")
        memory_length = st.slider(
            'Conversation Memory (messages)',
            1, 10, 5,
            help="Number of previous messages to remember"
        )
        
        # Persona selection
        st.subheader("AI Persona")
        st.session_state.selected_persona = st.selectbox(
            'Select conversation style:',
            ['Default', 'Expert', 'Creative']
        )

        # Chat statistics
        display_chat_statistics()

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.start_time = None
            st.session_state.user_input = ""
            st.rerun()


    # Main chat interface
    st.title("🤖 Groq Chat Assistant")   
    
    # Initialize chat components
    memory = ConversationBufferWindowMemory(k=memory_length)
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )
    
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=get_custom_prompt()
    )

    # Load chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            st.write(f"👤 You")
            st.info(message['human'])
        
        with st.container():
            st.write(f"🤖 Assistant ({st.session_state.selected_persona} mode)")
            st.success(message['AI'])

            # Audio response
            audio = text_to_audio(message['AI'])
            st.audio(audio, format="audio/mp3")

        st.write("")
    
    # User input section
    st.markdown("### 💭 Your Message")

    # Capture user input
    user_input = st.text_area(
        "",
        placeholder="Type your message here... (Shift + Enter to send)",
        key="user_input",
        help="Type your message and press Shift + Enter or click the Send button"
    )

   # Button section below the input box
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        if st.button("🎙️ Use Voice Input"):
            voice_input = get_voice_input()
            if voice_input:
                with st.spinner('🤔 Thinking...'):
                    try:
                        response = conversation(voice_input)
                        st.session_state.chat_history.append({
                            'human': voice_input,
                            'AI': response['response']
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

    with col2:
        if st.button("📤 Send") and user_input.strip():
            if not st.session_state.start_time:
              st.session_state.start_time = datetime.now()  # Set start time when the chat begins
            with st.spinner('🤔 Thinking...'):
                try:
                    response = conversation(user_input)
                    st.session_state.chat_history.append({
                        'human': user_input,
                        'AI': response['response']
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

    with col3:
        if st.button("🔄 New Topic"):
            memory.clear()
            st.success("Memory cleared for new topic!") 

    # Footer
    st.markdown("---")
    st.markdown(
        "Using Groq AI with "
        f"{st.session_state.selected_persona.lower()} persona | "
        f"Memory: {memory_length} messages"
    )
    st.markdown("""
      <div>
        <p style="color:#777770; font-size: 18px;">🤖 Built by Kunal Pusdekar 🚀</p>
      </div>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
