# Zepto AI Platform

An end-to-end AI and Data Engineering platform consisting of three independent modules:

- Data Pipeline
- Analytics & Machine Learning Pipeline
- Retrieval-Augmented Generation (RAG) Support Assistant

This repository contains the complete implementation of the **Zepto AI Platform Assignment**, demonstrating web scraping, data engineering, machine learning, semantic search, and AI-powered customer support.

---

# Repository Structure

```
zepto-ai-platform/
│
├── data_pipeline/
├── analytics/
├── support_assistant/
├── README.md
└── .gitignore
```

---

# Module 1 – Data Pipeline

## Objective

Develop a complete ETL pipeline by scraping book data, cleaning and transforming it, storing it in a relational database, and querying it using SQL and Pandas.

## Features

- Web scraping using Requests and BeautifulSoup
- Scraped books from multiple categories
- Data cleaning and preprocessing
- Price conversion from GBP to INR
- Normalized SQLite database
- Primary Key and Foreign Key relationships
- SQL queries for analysis
- Pandas integration with SQLite
- DataFrame merge validation

## Technologies

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite

---

# Module 2 – Analytics Pipeline

## Objective

Perform exploratory data analysis and develop machine learning models using the Titanic dataset.

## Dataset

The project uses the Titanic dataset loaded through Seaborn, with an offline copy included in the repository.

## Exploratory Data Analysis

- Dataset overview
- Missing value analysis
- Data cleaning
- Outlier detection
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Feature standardization

## Machine Learning

Implemented models include:

- Logistic Regression
- Decision Tree
- Random Forest

Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score

Additional tasks:

- SMOTE for class imbalance
- Class weight comparison
- GridSearchCV
- Out-of-Bag (OOB) evaluation
- Linear Regression
- Residual analysis

## Model Persistence

The best-performing pipeline is saved as:

```
analytics/models/best_pipeline.joblib
```

---

# Module 3 – Support Assistant

## Objective

Develop a Retrieval-Augmented Generation (RAG) support assistant capable of answering Zepto policy-related queries.

## Features

- Semantic document retrieval
- ChromaDB vector database
- Sentence Transformer embeddings
- LangGraph workflow
- Intent classification
- Prompt engineering
- FastAPI REST API
- Docker deployment
- Offline MOCK LLM mode

---

# System Workflow

```
User Query
      │
      ▼
Intent Classification
      │
      ├───────────────┐
      ▼               ▼
Policy Query     General Query
      │               │
      ▼               ▼
Retrieve        Direct Response
Documents
      │
      ▼
Prompt Construction
      │
      ▼
MOCK LLM
      │
      ▼
JSON Response
```

---

# Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Web Framework | FastAPI |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Imbalanced Learning | SMOTE |
| Web Scraping | Requests, BeautifulSoup |
| Database | SQLite |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Workflow | LangGraph |
| Containerization | Docker |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/zepto-ai-platform.git
```

Navigate to the project directory:

```bash
cd zepto-ai-platform
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Module 1

```bash
cd data_pipeline
python scrape_books.py
```

---

# Running Module 2

```bash
cd analytics
python eda.py
python modeling.py
```

Outputs generated include:

- Data visualizations
- Model comparison
- Residual plots
- Trained model pipeline (`best_pipeline.joblib`)

---

# Running Module 3

Build the vector database:

```bash
cd support_assistant
python ingest.py
```

Run the FastAPI application:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Docker

Build the Docker image:

```bash
docker build -t zepto-support .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support
```

---

# Project Highlights

- End-to-end ETL pipeline
- Web scraping and data cleaning
- Relational database design
- SQL and Pandas integration
- Exploratory Data Analysis
- Machine learning model comparison
- Hyperparameter optimization
- Saved production-ready ML pipeline
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- LangGraph workflow orchestration
- FastAPI REST API
- Dockerized deployment

---

# Future Enhancements

- Integration with OpenAI or Gemini APIs
- Hybrid keyword and semantic search
- Multi-language support
- Authentication and authorization
- Conversation memory
- Feedback collection
- Streaming API responses
- CI/CD pipeline
- Kubernetes deployment

---

# Author

**Manasvi P**

Bachelor of Technology (Computer Science)

---

# License

This project was developed for educational purposes as part of the **Zepto AI Platform Assignment**.