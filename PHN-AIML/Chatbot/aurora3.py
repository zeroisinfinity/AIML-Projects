import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

# -------------------------------
# 1. Setup & Configuration
# -------------------------------
dotenv_path = os.path.join(os.getcwd(), 'config', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Initialize Clients
# Official Google GenAI SDK (Required: pip install -U google-genai)
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# OpenRouter for GPT-4o Mini (Using your sk-or-v1 key)
open_router_client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Groq for high-speed Llama 3.1
llama_client = OpenAI(
    api_key=os.getenv("LLAMA_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title='AURORA Chatbot', page_icon='🤖')
st.title('AURORA: 2026 Stable Build')

# -------------------------------
# 2. Sidebar Settings
# -------------------------------
with st.sidebar:
    st.title("Model Hub")
    model_choice = st.selectbox('Choose the Brain:', [
        "gemini-2.5-flash",  # Primary Stable Model
        "gemini-2.5-flash-lite",  # Fastest / High Quota
        "Llama 3.1 (Groq)",
        "OpenRouter Free"
    ])

    st.write("---")
    st.write("Voice Input:")
    audio_info = mic_recorder(
        start_prompt="Start Recording 🎤",
        stop_prompt="Stop Recording 🛑",
        key='mic'
    )

    if st.button('Clear history'):
        st.session_state.messages = []
        st.session_state.last_audio_ts = None
        st.rerun()

# -------------------------------
# 3. Session State & History
# -------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_audio_ts' not in st.session_state:
    st.session_state.last_audio_ts = None

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# -------------------------------
# 4. Input Handling (The "Ear")
# -------------------------------
prompt = None

# Process Voice only if it's a NEW recording
if audio_info and audio_info.get('id') != st.session_state.last_audio_ts:
    try:
        with st.spinner("Gemini is transcribing..."):
            trans_res = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Transcribe this audio exactly. If silence, return nothing.",
                    types.Part.from_bytes(data=audio_info['bytes'], mime_type="audio/wav")
                ]
            )
            prompt = trans_res.text.strip()
            st.session_state.last_audio_ts = audio_info.get('id')
    except Exception as e:
        st.error(f"Voice Error: {e}")

# Process Keyboard Input
chat_input = st.chat_input('What is on your mind?')
if chat_input:
    prompt = chat_input

# -------------------------------
# 5. Response Generation (The "Brain")
# -------------------------------
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        with st.chat_message('assistant'):
            with st.spinner(f"{model_choice} is responding..."):

                # --- GEMINI ROUTE ---
                if "gemini" in model_choice:
                    history = [
                        {"role": "model" if m["role"] == "assistant" else "user",
                         "parts": [{"text": m["content"]}]}
                        for m in st.session_state.messages[:-1]
                    ]
                    chat = gemini_client.chats.create(model=model_choice, history=history)
                    response = chat.send_message(prompt)
                    full_response = response.text

                # --- NON-GEMINI ROUTE ---
                else:
                    if model_choice == "Llama 3.1 (Groq)":
                        active_client, m_id = llama_client, "llama-3.1-8b-instant"
                    elif model_choice == "OpenRouter Free":
                        active_client, m_id = open_router_client, "openrouter/free"
                    else:
                        # Defaults to GPT-4o Mini via OpenRouter using your sk-or-v1 key
                        active_client, m_id = open_router_client, "openai/gpt-4o-mini"

                    res = active_client.chat.completions.create(
                        model=m_id,
                        messages=st.session_state.messages
                    )
                    full_response = res.choices[0].message.content

            # Final Render and Save
            st.markdown(full_response)
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})

    except Exception as e:
        st.error(f"Error ({model_choice}): {e}")