from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from ecobiteAgent import build_agent_graph

# Postgres / Supabase Imports
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import uvicorn
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- SECURITY SETUP ---
# This defines the name of the header we expect (e.g., in Postman or React Native)
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Get the real secret key from the server environment
SERVER_API_KEY = os.getenv("ECOBITE_API_KEY")

# Dependency function that runs before every request
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not SERVER_API_KEY:
        # Fail open or closed? Safer to fail closed if config is missing.
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

# We add 'dependencies=[Depends(get_api_key)]' to secure ALL endpoints globally
app = FastAPI(
    title="EcoBite Agent API", 
    version="1.5", 
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)] 
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    global agent_app
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        user_message = HumanMessage(content=request.message)
        
        output = agent_app.invoke(
            {"messages": [user_message]},
            config=config
        )
        
        last_message = output["messages"][-1]
        
        return ChatResponse(
            response=last_message.content,
            thread_id=request.thread_id
        )
    except Exception as e:
        print(f"🔥 Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    global agent_app
    try:
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
                
            formatted_messages.append({
                "id": str(getattr(msg, "id", "")),
                "type": msg_type,
                "message": msg.content,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            })

        return {"messages": formatted_messages}

    except Exception as e:
        print(f"⚠️ History Error: {e}")
        return {"messages": []}

if __name__ == "__main__":
    print("🚀 Starting server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)