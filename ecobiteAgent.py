from typing import Annotated, TypedDict, Sequence, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from prompts.main_reply_prompt import get_main_reply_prompt

load_dotenv()

# --- CONFIGS ---
@dataclass
class AgentConfig:
    """Centralized configuration"""
    model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.5

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    config: AgentConfig

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
    
    return [current_dateTime]

    @tool(descriptio=
    """
    Fetch the user inventory from supabase
    """)
    def _fetch_inventory():
        return

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
        system_prompt = SystemMessage(content=get_main_reply_prompt())
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