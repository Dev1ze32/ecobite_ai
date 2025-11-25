from typing import Annotated, TypedDict, Sequence, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma

from prompts.main_reply_prompt import get_main_reply_prompt

load_dotenv()
db_dir = "./chroma_db" # Fixed typo from 'chromba_db' to 'chroma_db' for consistency
collection_name = "ecobite_faq"

# --- CONFIGS ---
@dataclass
class AgentConfig:
    """Centralized configuration"""
    model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.5

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    config: AgentConfig

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
if os.path.exists(db_dir) and os.listdir(db_dir):
    print(f"--- Existing database found in {db_dir}. Loading... ---")
    
    # Just load the existing DB, do not create new embeddings
    vectorStore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# --- TOOLS ---
def create_tools(config: AgentConfig): 
    @tool(description=
        """
        Use this tool ONLY when the user explicitly asks for the 'current date', 'current time', 
        'today's date', or 'what day is it'.
        """)
    def current_dateTime():
        now = datetime.now()
        return now.strftime("%b %d, %Y %H:%M") # Simplified return
    

    @tool
    def ecobite_faq_retriever(query: str) -> str:
        """
        This tool searches and returns the information from the EcoBite FAQ document 
        to answer questions related to the app, features, donations, and usage.
        """
        docs = retriever.invoke(query)

        if not docs:
            return f"Didn't find relevant information about {query} in the EcoBite FAQ document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Source Document Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}")
        
        return "\n\n".join(results)

    return [current_dateTime, ecobite_faq_retriever]

# --- GRAPH BUILDER ---
# CRITICAL CHANGE: Added 'checkpointer' argument here
def build_agent_graph(config: AgentConfig = None, checkpointer = None):
    if config is None:
        config = AgentConfig()

    # 1. Setup Tools and Model
    tools = create_tools(config)
    model = ChatOpenAI(
        model_name=config.model_name,
        temperature=config.temperature
    ).bind_tools(tools)

    # 2. Define Agent Node
    def agent_node(state: AgentState):
        messages = state["messages"]
        
        # Ensure System Prompt is always present
        system_prompt = SystemMessage(content=get_main_reply_prompt(tools[1].name))
        all_messages = [system_prompt] + messages
        
        response = model.invoke(all_messages)
        return {"messages": [response]}

    # 3. Define Routing Logic
    def route_agent(state: AgentState) -> Literal["tools", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "tools"
        return "end"

    # 4. Build the Graph
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=tools))

    graph.add_edge(START, "agent")
    
    graph.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "agent")

    # 5. Persistence Strategy
    # CRITICAL CHANGE: Logic to handle external DB or fallback to RAM
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)