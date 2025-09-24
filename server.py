from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# Add the Text directory to Python path
text_dir = Path(__file__).parent / "Text"
if str(text_dir) not in sys.path:
    sys.path.insert(0, str(text_dir))

# Import your chatbot logic
from Text import chatbot_text

app = FastAPI(title="Kissan Mitra Chatbot API", version="1.0.0")

# Define request model
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Define response model (match get_response output)
class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

@app.get("/")
async def root():
    return {"message": "Kissan Mitra Chatbot API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Kissan Mitra Chatbot"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        reply = chatbot_text.get_response(request.message, session_id=request.session_id)
        
        # Ensure reply is a dict with expected keys
        if not isinstance(reply, dict):
            raise ValueError("Chatbot did not return a valid response object")
        
        return ChatResponse(**reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "_main_":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)