import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"
VOCAB_FILE = "vocabulary.txt"

# Fast LLM to extract clean names during ingestion
fast_llm = ChatOllama(model="llama3.2:1b", temperature=0)

def extract_vocab_from_page(text):
    """Asks the fast model to pull out the real project/entity names from a page."""
    prompt = f"""Extract 3-5 core unique nouns, project names, or specific topics from this text. 
    Return them ONLY as a comma-separated list. No explanations.
    Text: {text[:1500]}
    Keywords:"""
    try:
        response = fast_llm.invoke(prompt)
        return response.content.strip().split(",")
    except Exception:
        return []

def save_vocabulary(new_words):
    """Saves unique keywords to a local file."""
    existing_words = set()
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r") as f:
            existing_words = set(line.strip().lower() for line in f if line.strip())
            
    for word in new_words:
        clean_word = word.strip().strip("'\"-.*")
        if clean_word and len(clean_word) > 2:
            existing_words.add(clean_word.lower())
            
    with open(VOCAB_FILE, "w") as f:
        for word in sorted(existing_words):
            f.write(f"{word}\n")

def ingest_docs():
    print("--- Starting Ingestion ---")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Folder '{DATA_PATH}' not found!")
        return

    documents = []
    discovered_keywords = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"Reading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            file_docs = loader.load()
            
            if file_docs:
                # Extract terms from the first page (contains the real names/titles)
                keywords = extract_vocab_from_page(file_docs[0].page_content)
                discovered_keywords.extend(keywords)
                
            documents.extend(file_docs)
    
    if not documents:
        print("No PDF files found.")
        return

    # Save keywords to cheat sheet
    save_vocabulary(discovered_keywords)
    print(f"📝 Updated local dictionary file: {VOCAB_FILE}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_docs()