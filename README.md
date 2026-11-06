# Local Agentic Research Assistant (LARA)

LARA is a privacy-first, offline-capable AI research assistant designed to process and interrogate complex documents entirely on consumer hardware. 

Unlike traditional RAG (Retrieval-Augmented Generation) pipelines that rely on cloud-based LLMs, this project leverages a dual-engine local architecture with deterministic state-machine orchestration to ensure complete data privacy and continuous availability.

## Technical Architecture

The system is built on a multi-agent architectural pattern using **LangGraph** for flow control, combining fast routing models with deeper reasoning models.

### Dual-Engine LLM Setup
- **Fast Brain (`llama3.2:1b`)**: Used for rapid query parsing, relevancy grading, and vocabulary extraction. Its low latency allows it to act as an efficient router and evaluator without bottlenecking the system.
- **Smart Brain (`phi4-mini-reasoning` / `llama3.2:3b`)**: Handles complex reasoning and final answer generation. It synthesizes graded context blocks while adhering to strict factual constraints to prevent hallucinations and cross-contamination.

### Core Components
- **Orchestration**: `LangGraph` implements a state machine workflow (Preprocess -> Retrieve -> Grade -> Rewrite -> Generate). This allows the agent to self-correct and rewrite failed search queries if the initial retrieval yields irrelevant documents.
- **Dynamic Vocabulary Extraction**: During PDF ingestion (`ingestion.py`), the Fast LLM dynamically extracts core nouns and project names to build a local dictionary (`vocabulary.txt`). This context-aware dictionary is used to sanitize user input and fix typos before vector retrieval.
- **Vector Storage**: `ChromaDB` provides local, persistent vector storage for document embeddings, powered by `nomic-embed-text`.
- **Interface**: A responsive local UI built with `Streamlit`, featuring real-time token streaming and document vault management.

## Installation & Setup

1. **Prerequisites**: Ensure [Ollama](https://ollama.com/) is installed and running on your host machine.
2. **Pull Local Models**:
   ```bash
   ollama pull llama3.2:1b
   ollama pull llama3.2:3b
   ollama pull phi4-mini-reasoning
   ollama pull nomic-embed-text
   ```
3. **Environment Setup**:
   Clone the repository and install the dependencies in a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install streamlit langgraph langchain langchain-community langchain-chroma langchain-ollama pypdf
   ```

## Usage

### 1. Ingestion Pipeline
Place your target PDF documents in the `data/` directory, then run the offline ingestion pipeline:
```bash
python ingestion.py
```
This script will parse the PDFs, chunk the text, extract the dynamic vocabulary, and build the local vector indices in `chroma_db/`.

### 2. Interactive Interface
Start the application server to open the Streamlit interface:
```bash
streamlit run app.py
```
You can query the system naturally. The agent will show its workflow (sanitizing queries, retrieving context, and evaluating relevancy) before streaming the final response.
> **Performance Note:** Execution speed may vary depending on your hardware specifications. While running these models on a dedicated GPU provides optimal performance, execution on a CPU-only setup may result in slower processing times.
> 
## Security & Privacy
Because all models and vector stores run locally, no data is transmitted over the internet. The application is completely air-gapped once the base models are downloaded, making it suitable for processing sensitive or confidential documents.
