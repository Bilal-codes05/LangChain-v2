# serve.py
from fastapi import FastAPI, Request
from langserve import add_routes
from chatbot import with_message_history, get_session_history

app = FastAPI()

add_routes(
    app,
    with_message_history,
    path="/chat",
    config_keys=["session_id"]
)

# Optional: Custom POST route for debugging or testing
@app.post("/chat/invoke")
async def invoke_with_wrapper(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    config = {"configurable": {"session_id": body.get("configurable", {}).get("session_id", "default")}}

    response = await with_message_history.ainvoke(body, config=config)
    return {"content": response.content}
