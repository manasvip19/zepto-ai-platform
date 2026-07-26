from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import MOCK_LLM
from app.rag import retrieve


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state):

    query = state["query"].lower()

    if any(word in query for word in KEYWORDS):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state
def retrieve_and_answer(state):

    docs, ids, meta = retrieve(state["query"])

    snippet = docs[0][:200]

    if MOCK_LLM:
        answer = (
            f"Based on the retrieved context: {snippet}"
        )
    else:
        answer = "Real LLM response"

    state["answer"] = answer
    state["sources"] = ids
    state["confidence"] = 1.0

    return state
def direct_answer(state):

    if MOCK_LLM:
        answer = (
            "I can only answer questions about Zepto policies right now."
        )
    else:
        answer = "Real LLM response"

    state["answer"] = answer
    state["sources"] = []
    state["confidence"] = 1.0

    return state
def router(state):

    return state["intent"]

builder = StateGraph(GraphState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    router,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)

builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()