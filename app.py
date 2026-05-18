import streamlit as st
import os
import tempfile
from typing import List, TypedDict
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

VOCAB_FILE = "vocabulary.txt"

# ==========================================
# 1. UI SETUP
# ==========================================
st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Privacy-First Research Assistant")
st.markdown("All data stays on this device. No internet required.")

# ==========================================
# 2. INITIALIZE ENGINES
# ==========================================
@st.cache_resource
def load_engines():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # BALANCED OPTIMIZATION: k=4 captures enough narrative context without flooding the CPU
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    fast_llm = ChatOllama(model="llama3.2:1b", temperature=0)
    smart_llm = ChatOllama(model="llama3.2:3b", temperature=0) 
    
    return embeddings, vectorstore, retriever, fast_llm, smart_llm

embeddings, vectorstore, retriever, fast_llm, smart_llm = load_engines()

def get_stored_vocabulary():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "r") as f:
            return ", ".join([line.strip() for line in f if line.strip()])
    return "e-choupal, capsense yoga, case study, solar power"

def append_to_vocabulary(text_sample):
    prompt = f"Extract 3-5 main proper nouns or topic phrases from this text as a comma list: {text_sample[:1000]}"
    try:
        res = fast_llm.invoke(prompt)
        words = res.content.strip().split(",")
        existing = set()
        if os.path.exists(VOCAB_FILE):
            with open(VOCAB_FILE, "r") as f:
                existing = set(line.strip().lower() for line in f if line.strip())
        for w in words:
            cw = w.strip().strip("'\"-.*").lower()
            if cw: existing.add(cw)
        with open(VOCAB_FILE, "w") as f:
            for w in sorted(existing): f.write(f"{w}\n")
    except: pass

# ==========================================
# 3. GRAPH LOGIC
# ==========================================
class GraphState(TypedDict):
    question: str
    search_queries: List[str]
    documents: List
    generation: str
    loop_count: int

def clean_output(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def preprocess_query(state: GraphState):
    valid_vocabulary = get_stored_vocabulary()
    
    prompt = PromptTemplate(
        template="""You are an expert query parser. Your ONLY job is to clean up the user's input.
        1. Fix typos using this valid dictionary: [{db_vocab}].
        2. If the user asks multiple distinct questions, split them into a MAXIMUM of 2 or 3 core search sentences. 
        DO NOT extract single words. Keep them as full search phrases.
        
        Output ONLY the optimized search phrases, one per line. No bullets, no quotes.
        
        User input: {question}
        Sanitized queries:""",
        input_variables=["question", "db_vocab"],
    )
    
    chain = prompt | smart_llm | StrOutputParser()
    raw_output = clean_output(chain.invoke({"question": state["question"], "db_vocab": valid_vocabulary}))
    queries = [q.strip() for q in raw_output.split("\n") if q.strip()]
    
    clean_bulleted_list = "\n".join([f"- {q}" for q in queries])
    return {"search_queries": queries, "question": clean_bulleted_list, "documents": [], "loop_count": 0}

def retrieve(state: GraphState):
    queries = state["search_queries"]
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))
        
    seen_content = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
            
    return {"documents": unique_docs, "question": state["question"], "search_queries": queries}

def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    # HYBRID BALANCED GRADER: Lenient enough to accept case histories and narrative themes, 
    # but strict enough to throw away 100% irrelevant files.
    prompt = PromptTemplate(
        template="""You are a data relevancy evaluator.
        Document content: {context}
        User Query Targets: {question}
        
        Does this document contain information, names, background, or themes relevant to AT LEAST ONE of the user query targets?
        If there is any thematic overlap or useful context, reply 'yes'. If it is completely off-topic, reply 'no'.
        Reply ONLY with the single word 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    grader_chain = prompt | fast_llm | StrOutputParser()
    filtered_docs = [d for d in documents if "yes" in clean_output(grader_chain.invoke({"question": question, "context": d.page_content})).lower()]
    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: GraphState):
    prompt = PromptTemplate(
        template="Rewrite this failed search query into a single short keyword phrase. Query: {question}\nKeywords:",
        input_variables=["question"],
    )
    rewrite_chain = prompt | fast_llm | StrOutputParser()
    new_q = clean_output(rewrite_chain.invoke({"question": state["question"]}))
    
    current_loops = state.get("loop_count", 0)
    return {"search_queries": [new_q], "question": state["question"], "documents": state["documents"], "loop_count": current_loops + 1}

def generate_answer(state: GraphState):
    return state

def decide_to_generate(state: GraphState):
    if not state["documents"]:
        if state.get("loop_count", 0) >= 1:
            return "generate"
        return "rewrite_query"
    else:
        return "generate"

# GRAPH LAYOUT
workflow = StateGraph(GraphState)
workflow.add_node("preprocess_query", preprocess_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("preprocess_query")
workflow.add_edge("preprocess_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"rewrite_query": "rewrite_query", "generate": "generate_answer"})
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)
app = workflow.compile()

# ==========================================
# 4. SIDEBAR: PDF UPLOAD
# ==========================================
with st.sidebar:
    st.header("📂 Document Vault")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing into secure vault..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    if docs:
                        append_to_vocabulary(docs[0].page_content)
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
                    chunks = text_splitter.split_documents(docs)
                    vectorstore.add_documents(chunks)
                    os.remove(tmp_file_path)
                st.success("✅ Documents indexed into local database!")
        else:
            st.warning("Please upload a PDF first.")

# ==========================================
# 5. CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        with st.status("Agent is researching...", expanded=True) as status:
            inputs = {"question": prompt_input}
            final_state = None
            
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "generate_answer":
                        final_state = value
                    elif key == "preprocess_query":
                        st.write(f"🧼 Sanitized input query using reference context parameters:")
                        for sq in value['search_queries']:
                            st.write(f"  👉 *\"{sq}\"*")
                    elif key == "retrieve":
                        st.write(f"📄 Retrieved {len(value['documents'])} context fragments matching parameters.")
                    elif key == "grade_documents":
                        st.write(f"⚖️ Evaluating relevancy thresholds. Retained {len(value['documents'])} verified blocks.")
            
            status.update(label="Research complete!", state="complete", expanded=False)

        with st.spinner("✍️ Local model is compiling response..."):
            if final_state and final_state.get("documents"):
                context = "\n\n".join([doc.page_content for doc in final_state["documents"]])
                working_prompt = final_state["question"]
            else:
                context = "No direct source documents available."
                working_prompt = prompt_input

            # FIX FOR INCONSISTENCY: Hardened system prompt constraints.
            # Even if the lenient grader lets a messy chunk slip by, this strict instruction
            # forces the smart 3B model to sort out the background noise and prevent data pollution.
            prompt_template = ChatPromptTemplate.from_messages([
                (
                    "system", 
                    "You are an expert academic evaluator. Your task is to provide a structured, factual answer based ONLY on the provided Context Blocks.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Evaluate the incoming text chunks carefully. Sift out any irrelevant background noise or cross-talk from unrelated documents.\n"
                    "2. If a specific topic cannot be explicitly proven or answered by the context blocks, write: 'The provided documents do not contain direct information regarding this topic.'\n"
                    "3. Do not make up facts, cross-contaminate data points, or blend separate case study domains together. Keep answers clearly separated by topic."
                ),
                (
                    "human", 
                    "Context Blocks:\n{context}\n\n"
                    "Objectives to resolve:\n{clean_query}\n\n"
                    "Detailed Factual Answer:"
                )
            ])
            
            generate_chain = prompt_template | smart_llm | StrOutputParser()
            output_placeholder = st.empty()
            full_response = ""
            
            for chunk in generate_chain.stream({"clean_query": working_prompt, "context": context}):
                full_response += chunk
                output_placeholder.markdown(full_response + "▌")
                
            output_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})