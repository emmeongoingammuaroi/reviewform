"""LangGraph workflow builder — wires all nodes into a review pipeline.

KEY CONCEPT — Graph Construction:
LangGraph uses a StateGraph to define the workflow. You:
1. Create a StateGraph with your state type (ReviewState)
2. Add nodes — each is a named function
3. Add edges — define the flow between nodes
4. Optionally add conditional edges — branching logic
5. Compile the graph — produces a runnable that you .ainvoke()

The compiled graph is a LangChain Runnable, so it supports:
- .ainvoke(state) — run the full pipeline
- .astream(state) — stream intermediate states
- Checkpointing — save/resume state (for human-in-the-loop)

Node list (6 nodes, each with a clear job):
- fetch_diff:          calls MCP -> GitHub, writes diff to state
- retrieve_standards:  calls MCP -> Qdrant, writes standards to state
- review_code:         GPT-4o: diff + standards -> structured feedback
- format_response:     formats final summary with severity counts
- human_review:        interrupt() -> wait -> resume
- log_eval:            writes session + scores to PostgreSQL

IMPORTANT — Dependency flow:
    agent nodes -> MCP client -> (SSE) -> mcp_server -> services/
    NOT: agent nodes -> services/  (that would bypass MCP entirely)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.nodes.fetch_diff import fetch_diff
from app.agent.nodes.format_response import format_response
from app.agent.nodes.human_review import human_review
from app.agent.nodes.log_eval import log_eval
from app.agent.nodes.retrieve_standards import retrieve_standards
from app.agent.nodes.review_code import review_code
from app.agent.state import ReviewState


def should_continue_after_human(state: ReviewState) -> str:
    """Conditional edge: decide what happens after human review.

    KEY CONCEPT — Conditional Edges:
    Unlike fixed edges (A -> B), conditional edges let you branch based on
    the current state. The function returns the name of the next node.
    This is how you implement if/else logic in your agent workflow.
    """
    if state.get("human_approved") is True:
        return "log_eval"  # Human approved — log the eval and finish
    elif state.get("human_approved") is False:
        return "review_code"  # Human rejected — re-review with their comments
    else:
        return END  # No human input yet — return current state (paused)


def build_review_graph() -> StateGraph:
    """Build and compile the code review workflow graph.

    The flow:
        fetch_diff -> retrieve_standards -> review_code -> format_response -> human_review
                                                ^                                |
                                                +--- (if rejected) --------------+
                                                                                 |
                                                              (if approved) -> log_eval -> END

    Returns a compiled graph that can be invoked with .ainvoke(initial_state).
    """
    # 1. Create the graph with our state type
    workflow = StateGraph(ReviewState)

    # 2. Add nodes — each node name maps to a function
    workflow.add_node("fetch_diff", fetch_diff)
    workflow.add_node("retrieve_standards", retrieve_standards)
    workflow.add_node("review_code", review_code)
    workflow.add_node("format_response", format_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("log_eval", log_eval)

    # 3. Define the flow with edges
    workflow.set_entry_point("fetch_diff")
    workflow.add_edge("fetch_diff", "retrieve_standards")
    workflow.add_edge("retrieve_standards", "review_code")
    workflow.add_edge("review_code", "format_response")
    workflow.add_edge("format_response", "human_review")

    # 4. Conditional edge after human review
    workflow.add_conditional_edges(
        "human_review",
        should_continue_after_human,
        {
            "log_eval": "log_eval",
            "review_code": "review_code",
            END: END,
        },
    )

    # 5. log_eval is the final step — it writes to PostgreSQL and ends
    workflow.add_edge("log_eval", END)

    # 6. Compile with a checkpointer for human-in-the-loop
    # MemorySaver stores state in memory. For production, use PostgresSaver.
    checkpointer = MemorySaver()
    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # Pause here for human input
    )

    return compiled


# Singleton graph instance — reused across requests
review_graph = build_review_graph()
