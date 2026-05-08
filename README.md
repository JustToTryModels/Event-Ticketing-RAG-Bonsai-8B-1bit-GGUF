<div align="center">

# Building a RAG Chatbot with Bonsai-8B (1-bit GGUF) for Domain-Specific QA

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/prism-ml/Bonsai-8B-gguf)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-blue)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-MiniLM-green)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

</div>

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline powered by **Bonsai-8B**—an end-to-end 1-bit language model by Prism ML. It demonstrates how to ground a compact, ultra-efficient LLM in external domain knowledge (event ticketing) using semantic retrieval, enabling accurate, context-aware responses without any fine-tuning.

The primary focus is the **RAG methodology and architecture**, showcasing how a 1.15 GB model combined with a vector store can achieve domain-specific performance that rivals larger, fine-tuned models.

<br>

---

## 📜 Table of Contents

1.  [**Core Architecture: The RAG Stack**](#1-core-architecture-the-rag-stack)
2.  [**The RAG Workflow**](#2-the-rag-workflow)
3.  [**Project Structure**](#3-project-structure)
4.  [**How to Run This Project**](#4-how-to-run-this-project)
5.  [**Results: In-Domain vs. Out-of-Domain**](#5-results-in-domain-vs-out-of-domain)
6.  [**License**](#6-license)

<br>

---

## 1. Core Architecture: The RAG Stack

This project implements a Naive RAG pipeline using a carefully selected stack of open-source tools, chosen for efficiency and compatibility.

<div align="center">

```
┌──────────────────────────────────────────────────────┐
│                    RAG PIPELINE                       │
│                                                      │
│   ┌─────────────────┐     ┌──────────────────────┐  │
│   │  Embedding Model │     │   Vector Store (DB)  │  │
│   │  all-MiniLM-L6-v2│────▶│       FAISS          │  │
│   │  (384-dim)       │     │   (IndexFlatL2)      │  │
│   └─────────────────┘     └──────────┬───────────┘  │
│                                      │ Top-K Chunks   │
│                                      ▼               │
│                           ┌──────────────────────┐   │
│                           │  Prompt Augmentation  │   │
│                           │  (Context + Query)    │   │
│                           └──────────┬───────────┘   │
│                                      │               │
│                                      ▼               │
│                           ┌──────────────────────┐   │
│                           │   LLM (Generator)    │   │
│                           │   Bonsai-8B (1-bit)  │   │
│                           │   via llama-cpp-python│   │
│                           └──────────┬───────────┘   │
│                                      │               │
│                                      ▼               │
│                           ┌──────────────────────┐   │
│                           │   Grounded Response   │   │
│                           └──────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

</div>

<br>

### 🧠 The Generator: Bonsai-8B (1-bit GGUF)

**Bonsai-8B** is an end-to-end 1-bit language model by Prism ML, quantized from the **Qwen3-8B** architecture. By reducing weight precision to a single bit (GGUF Q1_0 format), it achieves extreme efficiency while maintaining competitive performance.

| Specification | Detail |
|:---|:---|
| **Parameters** | 8.19B (~6.95B non-embedding) |
| **Architecture** | Qwen3-8B (GQA, SwiGLU MLP, RoPE, RMSNorm) |
| **Deployed Size** | **1.15 GB** (14.2x smaller than 16.38 GB FP16) |
| **Speedup** | **6.2x faster** than FP16 on RTX 4090 |
| **Energy** | **4.1x lower** energy/token on RTX 4090 |
| **1-bit Coverage** | Embeddings, attention, MLP, LM head |
| **License** | Apache 2.0 |

> Despite being 1/14th the size, Bonsai-8B scores a **70.5 avg** across 6 benchmarks, making it competitive with full-precision 8B models like Llama 3.1 8B (67.1) and Mistral3 8B (71.0).

### 🔎 The Embedding Model: all-MiniLM-L6-v2
A lightweight BERT-based model (~80 MB) producing **384-dimensional** dense vectors. It provides an excellent speed-quality balance for semantic similarity search.

### 🗄️ The Vector Store: FAISS
**FAISS** (Facebook AI Similarity Search) using `IndexFlatL2` for exact brute-force L2 nearest-neighbor search over ~25K instruction-response vectors.

### ⚙️ The Inference Engine: llama-cpp-python
Python bindings for `llama.cpp` providing GPU-accelerated inference with native Q1_0 dequantization kernels.

<br>

---

## 2. The RAG Workflow

### Step 1: Environment Setup

```bash
pip install -q sentence-transformers faiss-cpu
pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Step 2: Data Preparation & Vector Store Construction

Clean the dataset and encode instructions into the FAISS vector store.

```python
from sentence_transformers import SentenceTransformer
import faiss

# Initialize embedder
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all instructions
embeddings = embedder.encode(df['instruction'].tolist(), convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]  # 384
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Store reference data for retrieval
rag_data_list = df[['instruction', 'response']].to_dict('records')
```

### Step 3: Loading the LLM

Load Bonsai-8B with full GPU offloading (`n_gpu_layers=-1`).

```python
from llama_cpp import Llama

llm = Llama(
    model_path=model_path,
    n_ctx=4096,          # Context window
    n_gpu_layers=-1,     # Offload all layers to GPU
    verbose=False
)
```

### Step 4: Retrieval Logic

Encode the user query and find the top-K most similar instructions.

```python
def get_relevant_context(user_query, top_k=3):
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    contexts = []
    for idx in indices[0]:
        contexts.append(rag_data_list[idx]['response'])
    return contexts
```

### Step 5: RAG Inference

Combine retrieval and generation into the full pipeline.

```python
def rag_inference(user_query, top_k=3):
    # 1. Retrieve
    contexts = get_relevant_context(user_query, top_k)
    context_text = "\n\n".join(contexts)
    
    # 2. Augment
    prompt = f"""Use ONLY the following context to answer the question.
    If you cannot answer from the context, say "I don't have that information."
    Context: {context_text}
    Question: {user_query}
    Answer:"""
    
    # 3. Generate
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5, top_p=0.85, top_k=20, max_tokens=512
    )
    return response['choices'][0]['message']['content']
```

<br>

---

## 3. Project Structure

```bash
Event-Ticketing-Chatbot-using-RAG-and-Bonsai-8B/
│
├── Data/
│   └── Bitext-events-ticketing-llm-chatbot-training-dataset.csv
│
├── Notebook/
│   └── Event_Ticketing_Chatbot_RAG_Bonsai_8B.ipynb
│
├── Images & GIF/
│
├── LICENSE
└── README.md
```

<br>

---

## 4. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<username>/Event-Ticketing-Chatbot-using-RAG-and-Bonsai-8B.git
    cd Event-Ticketing-Chatbot-using-RAG-and-Bonsai-8B
    ```

2.  **Install dependencies:**
    ```bash
    pip install -q sentence-transformers faiss-cpu
    pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
    ```

3.  **Run the Notebook:**
    Open the notebook in **Google Colab** (with a T4 GPU or better) or a local Jupyter environment and execute the cells sequentially. Bonsai-8B's 1.15 GB footprint fits entirely on virtually any modern GPU.

<br>

---

## 5. Results: In-Domain vs. Out-of-Domain

The RAG system successfully **grounds responses in retrieved context**, preventing hallucinations on domain-specific queries while correctly refusing out-of-domain questions.

### ✅ In-Domain Query
> **User:** "How can I cancel my ticket for the upcoming cricket event in India?"
> 
> **RAG Assistant:** "To cancel your ticket for the upcoming event, please follow these steps:
> 1. Access the website and sign in to your account.
> 2. Go to the Ticket Cancellation section.
> 3. Locate your scheduled events and choose the event.
> 4. Select the Cancel Ticket option.
> 5. Complete the prompts to finalize your cancellation."

### ❌ Out-of-Domain Query
> **User:** "Explain the working of Large Language Models."
> 
> **RAG Assistant:** "I don't have that information. If you have questions about event ticketing, please let me know."

<br>

---

## 6. License

This project is licensed under the [MIT License](LICENSE).

**Note:** This project uses the **Bonsai-8B** model by Prism ML, licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0). Any use of the model must comply with its original license.

### Citation

If you use 1-bit Bonsai 8B, please cite:

```bibtex
@techreport{bonsai8b,
    title   = {1-bit Bonsai 8B: End-to-End 1-bit Language Model Deployment
               Across Apple, GPU, and Mobile Runtimes},
    author  = {Prism ML},
    year    = {2026},
    month   = {March},
    url     = {https://prismml.com}
}
```
