# frontend.py
import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Kissan Mitra Chatbot", page_icon="🌾", layout="centered")

st.title("🌾 Kissan Mitra Chatbot")
st.write("Talk to your AI assistant for farmers!")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call FastAPI server
    try:
        response = requests.post(API_URL, json={"message": user_input, "session_id": "user1"})
        if response.status_code == 200:
            data = response.json()
            reply = data.get("response", "No response from bot")
        else:
            reply = f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        reply = f"⚠ Could not connect to server: {e}"

    # Append bot reply to chat history
    st.session_state.messages.append({"role": "bot", "content": reply})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])