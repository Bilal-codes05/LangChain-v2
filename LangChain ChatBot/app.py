import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import with_message_history, get_session_history
import uuid

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")

st.title("🤖 LangChain ChatBot with Memory")
st.markdown("Ask anything, and I will remember!")

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Language Selector
language = st.selectbox("🌐 Select language for response:", ["English", "Spanish", "Urdu"], index=0)

# Optional: Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Type your message...")

# Handle user input
if user_input:
    human_msg = HumanMessage(content=user_input)
    st.session_state.chat_history.append(human_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        config = {"configurable": {"session_id": st.session_state.session_id}}
        with st.spinner("Thinking..."):
            response = ""
            response_box = st.empty()

            for chunk in with_message_history.stream(
                {
                    "messages": st.session_state.chat_history,
                    "language": language,
                },
                config=config,
            ):
                response += chunk.content
                response_box.markdown(response + "▌")

            response_box.markdown(response)

        ai_msg = AIMessage(content=response)
        st.session_state.chat_history.append(ai_msg)

# Show current session ID (optional)
st.caption(f"Session ID: `{st.session_state.session_id}`")
