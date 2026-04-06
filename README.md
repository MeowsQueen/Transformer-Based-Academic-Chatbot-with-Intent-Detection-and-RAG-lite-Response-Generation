# Transformer-Based-Academic-Chatbot-with-Intent-Detection

## Overview
This project presents a hybrid Natural Language Processing (NLP) system that combines **intent classification** and **retrieval-based question answering** to build an intelligent academic chatbot.

The system is designed to:
- understand user queries using an intent classification model
- retrieve relevant answers from a domain-specific knowledge base
- handle out-of-scope queries gracefully

This project was developed as part of the **Introduction to Large Language Models (LLM)** course.

---

## Project Structure
The system consists of two main components:

### 1. Intent Classification Module
- Trained on a subset of the **CLINC150 dataset**
- Performs multi-class classification of user queries
- Includes an **out-of-scope (OOS)** class for unknown inputs

### 2. Retrieval-Based Chatbot Module
- Uses a custom-built academic knowledge base
- Applies **transformer-based sentence embeddings**
- Retrieves the most semantically similar question using **cosine similarity**

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
    → Select relevant topic
    → Generate embedding (Sentence Transformers)
    → Compute similarity
    → Retrieve best matching answer
Else:
    → Return "Out of Scope" response
```
## Dataset

### Intent Classification Dataset

- Source: **CLINC150**
- Subset of **8 selected intents + 1 OOS class**

### Data Split

- Train: ~900 samples  
- Validation: ~260 samples  
- Test: ~480 samples  

### Knowledge Base (Retrieval)

- Custom-built dataset (~100 Q&A pairs)

#### Topics include:

- Transformers  
- BERT  
- LLMs  
- RAG  
- Embeddings  
- Attention  
- Tokenization  
- Fine-tuning  
- Course syllabus

## Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Sentence-Transformers  
- PyTorch  
- Streamlit  

---

## Methodology

### Preprocessing

- Lowercasing  
- Punctuation removal  
- Text normalization  

### Intent Classification

- Feature extraction using **TF-IDF**  
- Model: **Logistic Regression**  
- Training via gradient-based optimization  

### Retrieval System

- Sentence embeddings using pretrained transformer models  
- Similarity metric: **Cosine similarity**  

---

## Evaluation

### Classification Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### Retrieval Evaluation

- Top-1 Accuracy  
- Qualitative examples of chatbot responses  

---

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

- "What is BERT?"
- "Explain RAG"
- "How does attention work?"
- "When is the final exam?"
- "Tell me a joke" → Out of Scope

---

## Key Contributions

- Hybrid NLP system combining classification and retrieval
- Use of a real-world intent dataset (**CLINC150**)
- Integration of transformer-based embeddings
- Modular and extensible architecture

---

## Limitations

- The system cannot generate new answers (retrieval-only)
- Performance depends on knowledge base coverage
- Intent classifier trained on a limited dataset

---

## Future Work

- Integrate generative LLMs (e.g., GPT)
- Implement Retrieval-Augmented Generation (RAG)
- Improve intent classification using fine-tuned BERT
- Expand the knowledge base
