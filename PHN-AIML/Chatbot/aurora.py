import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from openai import OpenAI


# setup .env file path general to any os
dotenv_path = os.path.join(os.getcwd(),'config','.env')

# Load env file it takes key-val pairs n put in ur os prog process
load_dotenv(dotenv_path=dotenv_path)

# initialize client
gemini_client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
)
openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
)

# choose models industry gold std
GEMINI_MODEL = "gemini-flash-latest" # incredible speed, a large context window, and native multimodal support
OPENAI_MODEL = "gpt-4o-mini" # Industry-leading reasoning and reliable instruction following

# setup streamlit
st.set_page_config(
    page_title = 'AURORA Chatbot', # tab title
    page_icon = '🤖'
)

st.title('Your Personal Chatbot that remembers!') # H1 heading
st.write('ASK me anything from science to philosophy...') # paragraph below H1

# Toggle sidebar for choosing models
with st.sidebar:
    st.title("Settings")
    model_choice = st.selectbox(
        'Select Model',
        [GEMINI_MODEL,OPENAI_MODEL]
    )

    clear_history = st.button('Clear history')

# check if we've any memory
if 'messages' not in st.session_state:
    st.session_state.messages = []

# check if history was cleared
if clear_history:
    st.session_state.messages = []

# display history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# assign n truthy prompt
# walrus operator :=
if prompt := st.chat_input('What is in your mind?'):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user',
                                      'content':prompt})

    try:
        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                if model_choice == GEMINI_MODEL:
                    history = [
                        {
                            # This checks: if the role is assistant, call it "model" for Gemini
                            "role": "model" if m["role"] == "assistant" else "user",
                            "parts": [{"text": m["content"]}]
                        }
                        for m in st.session_state.messages[:-1]
                    ]

                    chat = gemini_client.chats.create(
                        model=GEMINI_MODEL,
                        history=history
                    )

                    response = chat.send_message(prompt)
                    full_response = response.text

                else:
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=st.session_state.messages
                    )

                    full_response = response.choices[0].message.content

            st.markdown(full_response)

        st.session_state.messages.append({
            'role': 'assistant',
            'content': full_response
        })
    except Exception as e:
        st.error(f"An error occurred: {e}")



