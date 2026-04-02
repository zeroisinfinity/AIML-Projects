import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

# -------------------------------
# 1. Setup & Configuration
# -------------------------------
dotenv_path = os.path.join(os.getcwd(), 'config', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Initialize Clients
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model Constants
GEMINI_MODEL = "gemini-flash-latest"
OPENAI_MODEL = "gpt-4o-mini"

st.set_page_config(page_title='AURORA Chatbot', page_icon='🤖')
st.title('Your Personal Chatbot that remembers!')
st.write('Ask me anything via text or voice...')

# -------------------------------
# 2. Sidebar Settings
# -------------------------------
with st.sidebar:
    st.title("Settings")
    model_choice = st.selectbox('Select Model', [GEMINI_MODEL, OPENAI_MODEL])

    st.write("Voice Input:")
    # The mic_recorder returns a dict when a recording is finished
    audio_info = mic_recorder(
        start_prompt="Start Recording 🎤",
        stop_prompt="Stop Recording 🛑",
        key='mic'
    )
    st.write("Debug Audio Info:", audio_info)

    st.write("---")
    clear_history = st.button('Clear history')

# -------------------------------
# 3. Session State & History
# -------------------------------
if 'messages' not in st.session_state or clear_history:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# -------------------------------
# 4. Handling Input (Voice or Text)
# -------------------------------
prompt = None

# Check for Voice Input first
if audio_info and 'text' in audio_info and audio_info['text']:
    prompt = audio_info['text']

# Check for Keyboard Input
chat_input = st.chat_input('What is on your mind?')
if chat_input:
    prompt = chat_input

# -------------------------------
# 5. AI Logic Block
# -------------------------------
if prompt:
    # Show User Message
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                if model_choice == GEMINI_MODEL:
                    # Format history for Gemini SDK
                    history = [
                        {
                            "role": "model" if m["role"] == "assistant" else "user",
                            "parts": [{"text": m["content"]}]
                        }
                        for m in st.session_state.messages[:-1]
                    ]

                    chat = gemini_client.chats.create(model=GEMINI_MODEL, history=history)
                    response = chat.send_message(prompt)
                    full_response = response.text

                else:
                    # OpenAI uses the standard messages list
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=st.session_state.messages
                    )
                    full_response = response.choices[0].message.content

            st.markdown(full_response)

        # Save Assistant Message
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")