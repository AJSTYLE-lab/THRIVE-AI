from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.documents import Document
import pickle
import re

app = FastAPI()

# ✅ Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define LangGraph state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ✅ Clean LLM response
def clean_response(text: str, question: str) -> str:
    patterns_to_remove = [
        r"(?i)^use the following context to answer.*?say so[.:]?\s*",  # Instruction
        re.escape(question) + r"[?:]?\s*",                             # Repeated question
        r"(?i)^human:\s*"                                             # Optional prefix
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text.strip()

# ✅ Load RAG system
def load_rag_components():
    tokenizer = AutoTokenizer.from_pretrained("./gpt2_model")
    model = AutoModelForCausalLM.from_pretrained("./gpt2_model", device_map="auto", torch_dtype=torch.float16)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)

    with open("./vector_store_config.pkl", "rb") as f:
        vector_store_config = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=vector_store_config["persist_directory"],
        embedding_function=embedding_model
    )

    with open("./prompt_template.pkl", "rb") as f:
        prompt_template = pickle.load(f)

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=3)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n".join(doc.page_content for doc in state["context"])
        messages = prompt_template.invoke({
            "question": state["question"],
            "context": docs_content
        })
        response = llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()

graph = load_rag_components()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    result = graph.invoke({"question": request.question})
    raw_answer = result.get("answer", "")
    cleaned_answer = clean_response(raw_answer, request.question)

    return {
        "question": request.question,
        "answer": cleaned_answer or "⚠️ No answer generated."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
