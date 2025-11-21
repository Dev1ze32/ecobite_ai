from typing import Annotated, TypedDict, Sequence, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os

from prompts.main_reply_prompt import get_main_reply_prompt

load_dotenv()

# CONFIGS
@dataclass
class AgentConfig:
    """Centralized configuration - easier to test and modify"""
    model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.5
    max_tokens: int = 500
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    config: AgentConfig

#TOOLS
def create_tools(cofig: AgentConfig):

    @tool(
        description=(
            "Use this tool ONLY when the user explicitly asks for the 'current date', 'current time', 'today's date', "
            "or 'what day is it'. This is crucial for context-aware responses, especially when comparing "
            "inventory expiration dates against the current calendar day."
        )
    )
    def current_dateTime():
        now = datetime.now()
        formatted_date = now.strftime("%b %d, %Y")
        formatted_time = now.strftime("%H:%M")
        return [formatted_date, formatted_time]
    
    return [current_dateTime]

# MAIN AGENT GRAPH
def build_agent_graph(config: AgentConfig) -> StateGraph:

    tools = create_tools(config)

    model = ChatOpenAI(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    ).bind_tools(tools)

    def agent_node(state: AgentState) -> AgentState:
        system_prompt = get_main_reply_prompt()
        # Initialize state
        state.setdefault("messages", [])
        messages = state["messages"]
        
        # If last message is a ToolMessage, AI responds without asking for input
        if messages and isinstance(messages[-1], ToolMessage):
            all_messages = [SystemMessage(content=system_prompt)] + messages
            
            try:
                response = model.invoke(all_messages)
                
                # Only print if there's actual content
                if response.content and response.content.strip():
                    print(f"\nAssistant: {response.content}")
                
                # Important: Only append AIMessage, not ToolMessages again
                return {"messages": [response]}
            except Exception as e:
                error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
                print(f"\nError: {str(e)}")
                return {"messages": [error_msg]}

        user_message = HumanMessage(content=input("\n💬 You: ").strip())
        all_messages = [SystemMessage(content=system_prompt)] + messages + [user_message]
        try:
            response = model.invoke(all_messages)
            # Only print if there's actual content
            if response.content and response.content.strip():
                print(f"\nAssistant: {response.content}")

            return {"messages": [user_message, response]}
        
        except Exception as e:
            error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
            print(f"\nError: {str(e)}")
            return {"messages": [user_message, error_msg]}
        
    def route_agent(state:AgentState) -> AgentState:
        messages = state.get("messages", [])
        last_message = messages[-1]

        if not messages:
            return "end"
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "use_tools"
        
        return "continue_chat"

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=tools))

    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    graph.add_conditional_edges(
        "agent",
        route_agent,
        {
            "use_tools": "tools",
            "continue_chat": "agent",
            "end": END
        }
    )
    graph.add_edge("tools", "agent")

    app = graph.compile()
    return app