from fastapi import FastAPI

from app.graph import graph
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0"
)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running."
    }


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):

    result = graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    })

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )