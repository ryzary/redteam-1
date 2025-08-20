from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict, List
import random

# Define state type
class ConversationState(TypedDict):
    history: List[str]
    turn: int
    max_turns: int

# Initialize LLM
# Change this to "gpt-4o-mini"
llm = ChatOllama(model="llama3:latest")

# Predefined follow-up prompts to create conversation
follow_up_prompts = [
    "That's interesting. Can you tell me more about that?",
    "I see your point. What would you suggest as an alternative approach?",
    "That makes sense. How do you think this applies to real-world situations?",
    "Thanks for explaining that. What are some common misconceptions about this topic?",
    "I appreciate your perspective. Can you give me a specific example?",
    "That's helpful. What would you recommend for someone just starting to learn about this?",
    "Interesting point. How has your understanding of this changed over time?",
    "I understand. What do you think are the most important aspects to consider?",
    "That's a good explanation. Are there any related topics you'd recommend exploring?",
    "Thanks for that insight. What questions do people usually ask about this subject?"
]

# Define the LLM node
def llm_node(state: ConversationState) -> ConversationState:
    # For the first turn, use the original message
    # For subsequent turns, create a conversational context
    if state["turn"] == 0:
        prompt = state["history"][-1]
    else:
        # Create a conversational prompt that includes the previous response
        previous_response = state["history"][-1]
        follow_up = random.choice(follow_up_prompts)
        prompt = f"Previous response: {previous_response}\n\nUser: {follow_up}"

    # Print the prompt being sent to the LLM
    print(f"\n--- PROMPT (Turn {state['turn'] + 1}) ---")
    print(prompt)

    # Call the LLM
    response = llm.invoke(prompt)

    # Update state with just the response (not the follow-up prompt)
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

# Define edges - conditional routing based on turn count
def continue_or_end(state: ConversationState) -> str:
    if state["turn"] >= state["max_turns"]:
        return "__end__"
    return "llm"

graph.add_conditional_edges("llm", continue_or_end)

# Set entrypoint
graph.set_entry_point("llm")

# Compile
app = graph.compile()

# Run the conversation
starter = "What are some effective ways to learn a new programming language?"
state = {"history": [starter], "turn": 0, "max_turns": 5}

for event in app.stream(state):
    for node_name, new_state in event.items():
        print(f"\n--- RESPONSE (Turn {new_state['turn']}) ---")
        print(new_state["history"][-1])
