from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from ecobiteAgent import build_agent_graph

# Postgres / Supabase Imports
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

# --- DATABASE SETUP ---
DB_URI = os.getenv("DATABASE_URL")

# We use a global variable to hold the compiled agent so we can access it in endpoints
agent_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager:
    1. Connects to Supabase when server starts.
    2. Sets up the tables if they don't exist.
    3. Compiles the agent with the DB connection.
    4. Closes connection when server stops.
    """
    global agent_app
    
    if DB_URI:
        masked_uri = DB_URI.replace(DB_URI.split(":")[2].split("@")[0], "******") if "@" in DB_URI else "Invalid URI format"
        print(f"🔌 Attempting to connect to: {masked_uri}")

        try:
            # --- CRITICAL FIXES FOR SUPABASE POOLER ---
            conn_kwargs = {
                "sslmode": "require",
                "autocommit": True,       # Fixes: "CREATE INDEX CONCURRENTLY cannot run inside a transaction block"
                "prepare_threshold": None # Fixes: Issues with Supabase Transaction Pooler (Port 6543)
            }
            
            pool = ConnectionPool(conninfo=DB_URI, max_size=15, kwargs=conn_kwargs)
            
            # Initialize the Postgres Saver
            checkpointer = PostgresSaver(pool)
            
            # This creates the tables. Now that autocommit is True, it should succeed.
            checkpointer.setup()
            
            print("✅ Connected to Supabase (Postgres)")
            agent_app = build_agent_graph(checkpointer=checkpointer)
            yield
            
            # Cleanup on shutdown
            pool.close()
            print("🛑 Disconnected from Supabase")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("⚠️ Falling back to in-memory storage (RAM).")
            agent_app = build_agent_graph()
            yield
    else:
        print("⚠️ No DATABASE_URL found. Using in-memory storage (RAM).")
        agent_app = build_agent_graph() 
        yield

# Initialize FastAPI with the lifespan manager
app = FastAPI(title="EcoBite Agent API", version="1.0", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/")
def read_root():
    return {"status": "EcoBite Agent is running"}

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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # CHANGED PORT TO 8001 to fix the "Address already in use" error
    print("🚀 Starting server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)