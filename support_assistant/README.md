# Zepto Support Assistant

> **AI-Powered Retrieval-Augmented Generation (RAG) Support Assistant for Zepto Policies**
>
> A production-ready support assistant built using **FastAPI, ChromaDB, Sentence Transformers, LangGraph, and Docker**. The assistant intelligently classifies user queries, retrieves relevant policy documents using semantic search, and generates grounded responses using a mock LLM workflow.

---

# Table of Contents

- Project Overview
- Objectives
- Features
- Technology Stack
- Project Structure
- System Architecture
- Workflow
- Retrieval-Augmented Generation (RAG)
- LangGraph Workflow
- Prompt Engineering
- Embedding & Vector Database
- API Endpoints
- Installation
- Running Locally
- Docker Deployment
- Example Requests
- Example Responses
- Project Modules
- Design Decisions
- Future Improvements
- Screenshots
- License

---

# Project Overview

The **Zepto Support Assistant** is an intelligent customer support chatbot that answers **only Zepto policy-related questions**.

Instead of using keyword matching, it performs **semantic retrieval** using Sentence Transformers and ChromaDB.

The application:

- accepts customer questions
- classifies whether they are policy questions
- retrieves relevant policy documents
- generates grounded answers
- rejects unrelated questions

This architecture follows the principles of **Retrieval-Augmented Generation (RAG)** while operating in **MOCK_LLM mode** for deterministic offline evaluation.

---

# Objectives

The project demonstrates:

- Semantic Search
- Vector Databases
- Retrieval-Augmented Generation
- LangGraph State Machines
- FastAPI REST APIs
- Prompt Engineering
- Docker Deployment
- AI System Design

---

# Features

## Semantic Document Search

Uses Sentence Transformers to search policies based on meaning instead of exact keywords.

---

## ChromaDB Vector Store

Stores document embeddings for efficient similarity search.

---

## Intent Classification

Classifies every incoming query as either

- Policy Question
- General Question

---

## LangGraph Workflow

Routes requests dynamically.

```
User Query
     │
     ▼
Intent Classification
     │
     ├──────────────┐
     ▼              ▼
Policy         General
Question       Question
     │              │
     ▼              ▼
Retrieve       Direct Reply
Context
     │
     ▼
Prompt
     │
     ▼
Mock LLM
     │
     ▼
Response
```

---

## Retrieval-Augmented Generation (RAG)

Instead of memorizing policies, the assistant retrieves relevant context before answering.

---

## Docker Support

Application can be deployed in a Docker container.

---

## Swagger Documentation

Interactive API documentation available at

```
http://localhost:7860/docs
```

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Framework | FastAPI |
| Workflow | LangGraph |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| API Validation | Pydantic |
| Containerization | Docker |
| Model | all-MiniLM-L6-v2 |
| Server | Uvicorn |

---

# Project Structure

```
support_assistant/
│
├── app/
│   ├── config.py
│   ├── graph.py
│   ├── main.py
│   ├── prompts.py
│   ├── rag.py
│   └── schemas.py
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── test_rag.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .env.example
```

---

# System Architecture

```
                    User Query
                         │
                         ▼
                 FastAPI Endpoint
                         │
                         ▼
                 LangGraph Workflow
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 Policy Question?                 General Question?
          │                             │
         YES                            NO
          │                             │
          ▼                             ▼
     Retrieve Context            Direct Response
          │
          ▼
      ChromaDB Search
          │
          ▼
   SentenceTransformer
          │
          ▼
     Prompt Generation
          │
          ▼
      MOCK LLM Response
          │
          ▼
        JSON Output
```

---

# Workflow

## Step 1

User submits a question.

Example

```
What is the delivery fee?
```

---

## Step 2

LangGraph determines whether the question is policy-related.

---

## Step 3

Relevant policy documents are retrieved from ChromaDB.

---

## Step 4

Prompt is constructed using

- role
- context
- task
- format
- response length

---

## Step 5

Mock LLM generates grounded response.

---

## Step 6

FastAPI returns

```json
{
    "answer": "...",
    "sources": [
        "doc_01",
        "doc_05"
    ],
    "confidence": 1
}
```

---

# Retrieval-Augmented Generation (RAG)

The assistant does **not** memorize Zepto policies.

Instead:

```
Question

↓

Embedding

↓

Vector Search

↓

Relevant Documents

↓

Prompt

↓

Generated Answer
```

Advantages

- Grounded responses
- Reduced hallucinations
- Easy document updates
- Scalable knowledge base

---

# LangGraph Workflow

State:

```
GraphState
```

Nodes

- classify_intent
- retrieve_and_answer
- direct_answer

Conditional Routing

```
Policy Question

↓

retrieve_and_answer

General Question

↓

direct_answer
```

---

# Prompt Engineering

Prompt template contains

## Role

You are a Zepto customer support assistant.

## Context

Retrieved policy documents.

## Task

Answer only using retrieved context.

## Format

Concise customer-friendly response.

## Length

Maximum 2-3 paragraphs.

## Negative Constraint

Do not answer using information not present in the retrieved context.

## Few-shot Example

Includes sample delivery-policy interaction.

---

# Embeddings

Model

```
all-MiniLM-L6-v2
```

Advantages

- Fast
- Lightweight
- High semantic accuracy
- Excellent retrieval performance

---

# Vector Database

Database

```
ChromaDB
```

Stores

- embeddings
- metadata
- document IDs

Retrieval

```
Top-K Similarity Search
```

---

# API Endpoints

## GET /

Returns

```
Zepto Support Assistant API
```

---

## POST /ask

Request

```json
{
    "query":"What is the delivery fee?"
}
```

Response

```json
{
    "answer":"Based on retrieved context...",
    "sources":[
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence":1
}
```

---

# Installation

Clone repository

```bash
git clone <repository-url>
```

Move into project

```bash
cd support_assistant
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build Vector Database

```bash
python ingest.py
```

Expected Output

```
Ingested 8 documents.
```

---

# Run Application

```bash
uvicorn app.main:app --reload
```

Visit

```
http://127.0.0.1:8000/docs
```

---

# Docker Deployment

Build

```bash
docker build -t zepto-support .
```

Run

```bash
docker run -p 7860:7860 zepto-support
```

Open

```
http://localhost:7860/docs
```

---

# Example Requests

## Policy Question

```json
{
    "query":"What is the delivery fee?"
}
```

Response

```json
{
    "answer":"Based on the retrieved context...",
    "sources":[
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence":1
}
```

---

## General Question

```json
{
    "query":"Who won the IPL?"
}
```

Response

```json
{
    "answer":"I can only answer questions about Zepto policies right now.",
    "sources":[],
    "confidence":1
}
```

---

# Project Modules

## Module 1

Document ingestion

- Read text files
- Generate embeddings
- Store vectors

---

## Module 2

Retrieval Engine

- Semantic Search
- Top-K Retrieval

---

## Module 3

Support Assistant

- LangGraph
- Prompt Construction
- Mock LLM
- FastAPI

---

# Design Decisions

Why ChromaDB?

- Lightweight
- Persistent
- Easy integration

---

Why Sentence Transformers?

- Excellent semantic embeddings
- Fast inference
- Open source

---

Why LangGraph?

- Explicit workflow
- State management
- Easy routing

---

Why FastAPI?

- High performance
- Automatic OpenAPI docs
- Pydantic validation

---

Why Mock LLM?

- Offline execution
- Deterministic outputs
- Reproducible evaluation

---

# Future Improvements

- OpenAI / Gemini integration
- Hybrid search
- Query rewriting
- Conversation memory
- Feedback collection
- Multi-language support
- Confidence calibration
- Authentication
- Streaming responses
- Real Zepto knowledge base
- Kubernetes deployment
- CI/CD pipeline

---

# Screenshots

Include screenshots of

- Swagger UI
- Docker container running
- Retrieval output
- ChromaDB ingestion
- API responses

---

# License

This project was developed for educational purposes as part of the **Zepto AI Platform – Support Assistant Module** using **FastAPI, ChromaDB, Sentence Transformers, LangGraph, and Docker**.