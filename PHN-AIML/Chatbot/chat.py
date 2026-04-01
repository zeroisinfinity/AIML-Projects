import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# -------------------------------
# Load API Key
# -------------------------------
# setup .env file path general to any os
dotenv_path = os.path.join(os.getcwd(),'config','.env')

# Load env file it takes key-val pairs n put in ur os prog process
load_dotenv(dotenv_path=dotenv_path)

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("models/gemini-flash-latest")

# -------------------------------
# Page UI
# -------------------------------
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖"
)

st.title("Your Personal Chatbot")
st.write("Ask me anything!")

# -------------------------------
# Chat History
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Show old messages
# -------------------------------
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# -------------------------------
# User Input
# -------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    {
                        "role": m["role"],
                        "parts": m["content"]
                    }
                    for m in st.session_state.messages[:-1]
                ]

                chat = model.start_chat(history=history)
                response = chat.send_message(user_input)

                ai_message = response.text

                st.markdown(ai_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_message
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")