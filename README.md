# Transformer-Based Academic Chatbot with Intent Detection and RAG-lite Response Generation

## Overview

This project presents a hybrid Natural Language Processing (NLP) system that combines **intent classification**, **retrieval-based question answering**, and **lightweight language generation** to build an intelligent academic chatbot.

The system is designed to:

* understand user queries using a trained intent classification model
* retrieve relevant information from a **domain-specific academic knowledge base**
* generate more natural and context-aware responses using a lightweight LLM-based generation step
* handle out-of-scope queries robustly

The chatbot focuses on **student-oriented academic queries**, including:

* course-related questions (schedule, grading, instructor)
* conceptual questions (e.g., NLP, transformers, embeddings)
* general academic assistance

This project was developed as part of the **Introduction to Large Language Models (LLM)** course.

---

## System Highlights

* Hybrid pipeline combining intent detection, retrieval, and generation
* Lightweight RAG-style architecture without full LLM dependency
* Query-type aware response generation (definition, comparison, why, etc.)
* Efficient and interpretable design suitable for academic applications

---

### Key Enhancements

Compared to a basic retrieval-based chatbot, this system includes:

* Query-aware reranking that aligns retrieved results with user intent
* Hybrid scoring combining semantic similarity and keyword grounding
* Structured knowledge base design with topic and subtopic hierarchy
* Improved handling of definition, comparison, and "why" questions
* Enhanced robustness for short queries (e.g., "llm", "rag")

---
## Project Structure

The system consists of three main components:

### 1. Intent Classification Module

* Trained on a subset of the **CLINC150 dataset**
* Performs multi-class classification of user queries
* Includes an **Out-of-Scope (OOS)** class to detect irrelevant queries
* Acts as a **general-purpose intent understanding layer**

### 2. Retrieval Module

* Uses a **custom-built academic knowledge base**
* Applies **transformer-based sentence embeddings**
* Retrieves the most semantically relevant question-answer pairs using **cosine similarity**
* Supplies contextual information for final response construction

### 3. RAG-lite Generation Module

* Takes the retrieved context and the user query as input
* Uses a lightweight language model or generation layer to produce a more natural answer
* Improves fluency and flexibility compared to returning a fixed stored answer
* Functions as a simplified form of **Retrieval-Augmented Generation (RAG)**

---

## Repository Structure

```text
Transformer-Based-Academic-Chatbot-with-Intent-Detection-and-RAG-lite-Response-Generation/
├── data/
│   ├── raw/
│   │   └── clinc150_full.json
│   │
│   └── processed/
│       ├── clinc_subset_train.csv
│       ├── clinc_subset_val.csv
│       ├── clinc_subset_test.csv
│       └── knowledge_base.csv
│
├── src/
│   ├── prepare_clinc.py
│   ├── preprocess.py
│   ├── train_intent_classifier.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── chatbot.py
│   └── app.py
│
├── models/
│   ├── intent_classifier.pkl
│   └── kb_embeddings.npy
│
├── results/
│   ├── metrics.txt
│   ├── confusion_matrix.png
│   └── sample_outputs.txt
│
├── notebooks/
│   └── experiments.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```
## Folder Descriptions

* **data/raw/**: original downloaded datasets  
* **data/processed/**: cleaned and prepared datasets used by the models  
* **src/**: source code for preprocessing, training, retrieval, and application logic  
* **results/**: saved evaluation outputs and visualizations  
* **notebooks/**: optional experiments and exploratory analysis  
* **requirements.txt**: Python dependencies  
* **README.md**: project documentation
  
---

## System Pipeline

```text
User Input
   ↓
Text Preprocessing
   ↓
Intent Classification (TF-IDF + Logistic Regression)
   ↓
If In-Scope:
    → Map intent to domain/topic
    → Generate embedding (Sentence Transformers)
    → Compute similarity
    → Retrieve top relevant context
    → Send retrieved context + user query to generation module
    → Return final generated response
Else:
    → Return "Out of Scope" response
```
## Dataset

### 1. Intent Classification Dataset

* **Source:** CLINC150  
* A subset of **8 selected intents + 1 OOS class**  
* Used to train a general intent classification model  

#### Data Split

* Train: ~900 samples  
* Validation: ~260 samples  
* Test: ~480 samples  

This dataset enables the system to handle **diverse natural language queries** and detect irrelevant inputs.

### 2. Knowledge Base (Retrieval Dataset)

* Custom-built dataset (~280+ Q&A pairs)
* Expanded using academic resources and course materials
* Includes both course-specific knowledge and transformer-based NLP concepts
* Designed specifically for this project  

#### Covered Domains

* Course and syllabus information  
* Exams and grading policies  
* Instructor and schedule queries  
* NLP and AI concepts (e.g., transformers, embeddings, BERT, GPT)  

The dataset includes:

* paraphrased questions  
* different linguistic variations  
* mixed query styles (formal + informal)  

This ensures more realistic chatbot behavior.

## Technologies Used

* Python  
* Pandas  
* Scikit-learn  
* Sentence-Transformers  
* PyTorch  
* HuggingFace Transformers  
* Streamlit  

## Methodology

### Preprocessing

* Lowercasing  
* Punctuation removal  
* Text normalization  

### Intent Classification

* Feature extraction using **TF-IDF**  
* **Model:** Logistic Regression  
* Trained using gradient-based optimization  

This module learns to map user queries into intent categories.

### Retrieval System

* Sentence embeddings generated using pretrained transformer models  
* Similarity computed using **cosine similarity**  
* Top relevant context is retrieved from the knowledge base  

This module performs semantic search over the academic corpus.

### RAG-lite Generation

* The retrieved context is combined with the user question  
* A lightweight generation step produces a fluent and context-aware response  
* This allows the chatbot to generate better natural-language answers instead of only returning a fixed stored response  

This module extends the system from a purely retrieval-based chatbot into a lightweight retrieval-augmented generation architecture.

## Evaluation

### Classification Metrics

* Accuracy  
* Precision  
* Recall  
* F1-score  
* Confusion Matrix  

### Retrieval Evaluation

* Top-k Retrieval Accuracy  
* Relevance of retrieved context  
* Qualitative inspection of retrieved matches  

### Generation Evaluation

* Qualitative analysis of answer fluency and coherence  
* Comparison between retrieved raw answer and generated final answer  
* Case-based evaluation on representative user queries

## System Behavior

The system includes safeguards to ensure reliable responses:

* **Out-of-Scope Detection:** Queries unrelated to the academic domain are identified and rejected.
* **Low-Confidence Retrieval Handling:** If the retrieved context has low similarity scores, the system avoids generating potentially incorrect answers and instead returns a safe fallback response.
* **Grounded Responses:** All answers are strictly based on retrieved knowledge base content.
  
## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the intent classifier

```bash
python src/train_intent_classifier.py
```

### 3. Run the chatbot

```bash
streamlit run src/app.py
```

## Example Queries

### Academic / Course Queries

* "When is the final exam?"
* "How is the course graded?"
* "Who is the instructor?"

### Conceptual Questions

* "What is a transformer?"
* "Explain embeddings in NLP"
* "What is the difference between BERT and GPT?"

### Out-of-Scope Example

* "Tell me a joke"

## Key Contributions

* Hybrid NLP system combining **intent classification, retrieval, and generation**  
* Use of a real-world dataset (**CLINC150**) for intent modeling  
* Integration of **transformer-based embeddings**  
* Extension of retrieval-based QA into a **RAG-lite architecture**  
* Domain-adapted academic chatbot design  
* Modular and extensible architecture  

## Limitations

* Performance depends on knowledge base coverage  
* Generated responses are only as strong as the retrieved context  
* The system is not a full-scale end-to-end RAG pipeline with vector database infrastructure  

## Future Work

* Implement full Retrieval-Augmented Generation (RAG) with vector database support  
* Replace the baseline classifier with fine-tuned BERT  
* Add confidence-based reranking for retrieved contexts  
* Expand the academic knowledge base  

## Notes

* The knowledge base embeddings (`kb_embeddings.npy`) are generated automatically at runtime and are not stored in the repository.
* The system prioritizes correctness over fluency by grounding responses in retrieved evidence.
