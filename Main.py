import re # We need this to clean up the output
from typing import List, TypedDict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

# --- 1. INITIALIZE DUAL ENGINES ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Fast Brain: For quick Yes/No grading and short rewrites
fast_llm = ChatOllama(model="llama3.2:1b", temperature=0)

# Smart Brain: For the final answer generation
smart_llm = ChatOllama(model="phi4-mini-reasoning", temperature=0)

class GraphState(TypedDict):
    question: str
    documents: List
    generation: str

def clean_output(text):
    """Helper to remove any <think> tags from the model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def retrieve(state: GraphState):
    print("\n---NODE: RETRIEVING DATA---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    print("\n---NODE: GRADING DOCUMENTS (Using Fast LLM)---")
    question = state["question"]
    documents = state["documents"]

    prompt = PromptTemplate(
        template="""You are a strict grader. 
        Document: {context} 
        Question: {question}
        Does the document contain keyword(s) or meaning related to the question? 
        Reply ONLY with 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    
    # Use the fast LLM here
    grader_chain = prompt | fast_llm | StrOutputParser()

    filtered_docs = []
    for d in documents:
        score = grader_chain.invoke({"question": question, "context": d.page_content})
        score = clean_output(score).lower()
        if "yes" in score:
            print("Grader: RELEVANT.")
            filtered_docs.append(d)
        else:
            print("Grader: IRRELEVANT.")
            
    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: GraphState):
    print("\n---NODE: REWRITING QUERY (Using Fast LLM)---")
    question = state["question"]

    prompt = PromptTemplate(
        template="""Write a better search query based on this failed one: {question}
        Reply ONLY with the new query keywords. No intro, no quotes.""",
        input_variables=["question"],
    )

    # Use the fast LLM here
    rewrite_chain = prompt | fast_llm | StrOutputParser()
    new_question = rewrite_chain.invoke({"question": question})
    new_question = clean_output(new_question)
    
    print(f"New Query: {new_question}")
    return {"question": new_question, "documents": state["documents"]}

def generate_answer(state: GraphState):
    print("\n---NODE: GENERATING ANSWER (Using Smart LLM)---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = PromptTemplate(
        template="""Answer the question based on the context. If you don't know, say "I don't know based on the documents."
        Question: {question}
        Context: {context}
        Answer:""",
        input_variables=["question", "context"],
    )

    # Use the Smart LLM here
    generate_chain = prompt | smart_llm | StrOutputParser()
    answer = generate_chain.invoke({"question": question, "context": context})
    answer = clean_output(answer)

    return {"generation": answer, "documents": documents, "question": question}

def decide_to_generate(state: GraphState):
    if not state["documents"]:
        return "rewrite_query"
    else:
        return "generate"

# --- 5. WIRING (Same as before) ---
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_generate, 
    {"rewrite_query": "rewrite_query", "generate": "generate_answer"}
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    # Test the tough question again
    test_question = "What was the exact financial budget allocated for marketing the CapSense Yoga mat in Q3 2023?"
    inputs = {"question": test_question}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass 
            
    print("\nFINAL ANSWER:\n", value["generation"])