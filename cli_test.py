from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
from cli_ai import AgentConfig
from cli_ai import build_agent_graph

def print_messages(messages):
    """Display tool results with full content"""
    if not messages:
        return
    
    # Get the most recent tool message
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            print(f"\n{'='*60}")
            print(f"🛠️ TOOL RESULT:")
            print(f"{'='*60}")
            print(message.content)  # Print FULL content, no truncation
            print(f"{'='*60}\n")
            break  # Only show the most recent tool result
        elif isinstance(message, AIMessage):
            print(f"\n{'='*60}")
            print(f"AI REPLY:")
            print(f"{'='*60}")
            print(message.content)  # Print FULL content, no truncation
            print(f"{'='*60}\n")

def run_ecobite_agent():
    """Enhanced CLI with better UX for ecoBite"""
    load_dotenv()
    config = AgentConfig()
    
    print("\n" + "=" * 70)
    print("        ♻️ ECOBITE - Intelligent Food Management Assistant")
    print("=" * 70)
    print("    I can help you track inventory, suggest meals, reduce waste, and save money.")
    print("    Type 'quit', 'exit', or 'bye' to end the session.")
    print("=" * 70 + "\n")
    
    app = build_agent_graph(config)
    state = {
        "messages": [],
        "config": config
    }
    
    try:
        for step in app.stream(state, stream_mode="values"):
            # Stream handles display, just track state
            state = step
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("         ⚠️  SESSION INTERRUPTED BY USER")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("\n" + "=" * 70)
        print("         ✓ ECOBITE SESSION ENDED")
        # Changed the output message to be more relevant to an inventory/impact app
        print(f"         Impact data or conversation log saved to: {config.output_dir}")
        print("=" * 70 + "\n")

run_ecobite_agent()