from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from ecobiteAgent import build_agent_graph

# Rate Limiting Imports (The "Speed Bump")
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Postgres / Supabase Imports
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import uvicorn
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- RATE LIMITER SETUP ---
# Identifies users by their IP address
limiter = Limiter(key_func=get_remote_address)

# --- SECURITY SETUP ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
SERVER_API_KEY = os.getenv("ECOBITE_API_KEY")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API Key configuration is missing"
        )
    if api_key_header != SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )
    return api_key_header

# --- DATABASE SETUP ---
DB_URI = os.getenv("DATABASE_URL")
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    if DB_URI:
        try:
            conn_kwargs = {
                "sslmode": "require",
                "autocommit": True,
                "prepare_threshold": None,
                "keepalives": 1,
                "keepalives_idle": 5,
                "keepalives_interval": 2,
                "keepalives_count": 5
            }
            
            pool = ConnectionPool(
                conninfo=DB_URI, 
                max_size=20, 
                min_size=0, 
                max_lifetime=300, 
                kwargs=conn_kwargs
            )
            
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            
            print("✅ Connected to Supabase (Postgres) with Robust Pool")
            agent_app = build_agent_graph(checkpointer=checkpointer)
            yield
            
            pool.close()
            print("🛑 Disconnected from Supabase")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("⚠️ Switching to IN-MEMORY storage")
            agent_app = build_agent_graph() 
            yield
    else:
        print("⚠️ No DATABASE_URL set. Using in-memory storage.")
        agent_app = build_agent_graph()
        yield

app = FastAPI(
    title="EcoBite Agent API", 
    version="1.5", 
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)] 
)

# Initialize Rate Limiter on App
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- STRICT INPUT VALIDATION ("The Weight Limit") ---
class ChatRequest(BaseModel):
    # Limit message to 2000 chars (approx 300-400 tokens)
    message: str = Field(..., max_length=2000, min_length=1)
    
    # Limit thread_id to prevent database key abuse
    thread_id: str = Field(..., max_length=100)
    
    # Added user_id so we can fetch specific inventory
    user_id: int

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute") # Strict: Only 5 chats per minute per IP
def chat_endpoint(request: Request, chat_req: ChatRequest):
    global agent_app
    try:
        # We pass user_id in the config metadata as well, which is good practice
        config = {
            "configurable": {
                "thread_id": chat_req.thread_id,
                "user_id": chat_req.user_id 
            }
        }
        
        # We also inject the user_id into the message content.
        # This ensures the LLM explicitly "sees" the ID in the context window
        # so it knows what argument to pass to 'user_inventory_retriever(user_id=...)'
        context_aware_message = f"User ID: {chat_req.user_id}\n\n{chat_req.message}"
        user_message = HumanMessage(content=context_aware_message)
        
        output = agent_app.invoke(
            {"messages": [user_message]},
            config=config
        )
        
        last_message = output["messages"][-1]
        
        return ChatResponse(
            response=last_message.content,
            thread_id=chat_req.thread_id
        )
    except Exception as e:
        print(f"🔥 Chat Error: {e}")
        # Be careful not to expose internal stack traces to users
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/history/{thread_id}")
@limiter.limit("20/minute") # Less strict for history, but still protected
def get_history(thread_id: str, request: Request):
    global agent_app
    try:
        # Validate thread_id length manually here since it's a path param
        if len(thread_id) > 100:
             raise HTTPException(status_code=422, detail="Thread ID too long")

        config = {"configurable": {"thread_id": thread_id}}
        state = agent_app.get_state(config)
        
        if not state.values:
            return {"messages": []}
            
        formatted_messages = []
        
        for msg in state.values.get("messages", []):
            msg_type = "user"
            if isinstance(msg, AIMessage):
                msg_type = "ai"
            elif isinstance(msg, HumanMessage):
                msg_type = "user"
            else:
                continue 
            
            # Clean up the message content for history display if needed
            # (Optional: strip the "User ID: ..." prefix if you don't want to show it in UI)
            content = msg.content
            if msg_type == "user" and content.startswith("User ID:"):
                # Simple split to hide the system injection from the frontend history if desired
                # But kept as-is for now for transparency/debugging
                pass

            formatted_messages.append({
                "id": str(getattr(msg, "id", "")),
                "type": msg_type,
                "message": content,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            })

        return {"messages": formatted_messages}

    except Exception as e:
        print(f"⚠️ History Error: {e}")
        return {"messages": []}

if __name__ == "__main__":
    print("🚀 Starting Secure Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)