import getpass
import os

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT_NAME"] = "ChatBot"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Import model
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Load model
model = ChatGroq(model="llama3-8b-8192")

# Session store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Set up prompt template with language support
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages")
])

# Trimmer setup
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Chain with trimming
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# Runnable with session memory
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# Session config
config = {"configurable": {"session_id": "abc15"}}

# User messages
messages = [
    SystemMessage(content="You're a good assistant"),
    HumanMessage(content="Hi! I'm Bilal Rafique."),
    AIMessage(content="Hi Bilal! How can I help you today?"),
    HumanMessage(content="Tell me a joke."),
]

# 🟢 Method 1: Clean final output after streaming
full_response = ""
for r in with_message_history.stream(
    {
        "messages": messages + [HumanMessage(content="Tell me another joke!")],
        "language": "English",
    },
    config=config,
):
    full_response += r.content

print("\nFull Response:")
print(full_response)

# 🟡 Method 2: (Optional) Live streaming with delay — comment out above and use this if needed
# import time
# for r in with_message_history.stream(
#     {
#         "messages": messages + [HumanMessage(content="Tell me another joke!")],
#         "language": "English",
#     },
#     config=config,
# ):
#     print(r.content, end="", flush=True)
#     time.sleep(0.01)
