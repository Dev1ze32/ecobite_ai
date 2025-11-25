from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from ecobiteAgent import build_agent_graph

# Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Postgres / Supabase Imports
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import uvicorn
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- RATE LIMITER SETUP (FIXED FOR RAILWAY/CLOUD) ---
def get_real_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # The first IP in the list is the real client
        return forwarded.split(",")[0]
    # Fallback to direct connection (useful for local testing)
    return request.client.host or "127.0.0.1"

limiter = Limiter(key_func=get_real_ip)

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
            # SSL Mode is required for Supabase/Cloud Postgres
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

# --- CORS MIDDLEWARE (Crucial for Web/Mobile access) ---
# Allows your specific frontend apps to talk to this server
app.add_middleware(
    CORSMiddleware,
    # Mobile apps often don't send an Origin header, so they bypass this check (which is fine).
    # Browsers DO send it, so this whitelist secures your web dashboard.
    allow_origins=["https://ecobite-website-m55hr2wbn-dell-ancisos-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- STRICT INPUT VALIDATION ---
class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000, min_length=1)
    thread_id: str = Field(..., max_length=100)
    user_id: int

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
def chat_endpoint(request: Request, chat_req: ChatRequest):
    global agent_app
    try:
        config = {
            "configurable": {
                "thread_id": chat_req.thread_id,
                "user_id": chat_req.user_id 
            }
        }
        
        # Inject user_id into context
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
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/history/{thread_id}")
@limiter.limit("20/minute")
def get_history(thread_id: str, request: Request):
    global agent_app
    try:
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
            
            content = msg.content
            # Optional: Hide system injections from history
            if msg_type == "user" and content.startswith("User ID:"):
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
    # RAILWAY REQUIREMENT:
    # Railway provides the PORT variable. If it's not found, default to 8001.
    port = int(os.getenv("PORT", 8001))
    print(f"🚀 Starting Secure Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)