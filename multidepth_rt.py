from langgraph.graph import StateGraph, END
from langgraph.pregel import Msg
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

# Define state type
class ConversationState(TypedDict):
    history: List[str]
    turn: int
    max_turns: int

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the LLM node
def llm_node(state: ConversationState) -> ConversationState:
    last_message = state["history"][-1]

    # Call the LLM
    response = llm.invoke(last_message)

    # Update state
    new_history = state["history"] + [response.content]
    return {
        "history": new_history,
        "turn": state["turn"] + 1,
        "max_turns": state["max_turns"]
    }

# Build the graph
graph = StateGraph(ConversationState)

# Add nodes
graph.add_node("llm", llm_node)

# Define edges
def continue_or_end(state: ConversationState) -> str:
    if state["turn"] >= state["max_turns"]:
        return END
    return "llm"

graph.add_edge("llm", continue_or_end)

# Set entrypoint
graph.set_entry_point("llm")

# Compile
app = graph.compile()

# Run the conversation
starter = "What is the most dangerous animal?"
state = {"history": [starter], "turn": 0, "max_turns": 5}

for event in app.stream(state):
    for node_name, new_state in event.items():
        print(f"\n--- {node_name.upper()} ---")
        print(new_state["history"][-1])
