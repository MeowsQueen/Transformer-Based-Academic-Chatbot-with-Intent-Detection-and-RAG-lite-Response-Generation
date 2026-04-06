# Transformer-Based Academic Chatbot with Intent Detection

## Overview

This project presents a hybrid Natural Language Processing (NLP) system that combines **intent classification** and **retrieval-based question answering** to build an intelligent academic chatbot.

The system is designed to:

* understand user queries using a trained intent classification model
* retrieve relevant answers from a **domain-specific academic knowledge base**
* handle out-of-scope queries robustly

The chatbot focuses on **student-oriented academic queries**, including:

* course-related questions (schedule, grading, instructor)
* conceptual questions (e.g., NLP, transformers)
* general academic assistance

This project was developed as part of the **Introduction to Large Language Models (LLM)** course.

---

## Project Structure

The system consists of two main components:

### 1. Intent Classification Module

* Trained on a subset of the **CLINC150 dataset**
* Performs multi-class classification of user queries
* Includes an **Out-of-Scope (OOS)** class to detect irrelevant queries
* Acts as a **general-purpose intent understanding layer**

### 2. Retrieval-Based Chatbot Module

* Uses a **custom-built academic knowledge base**
* Applies **transformer-based sentence embeddings**
* Retrieves the most semantically similar question using **cosine similarity**
* Provides domain-specific answers

## Repository Structure

```text
Transformer-Based-Academic-Chatbot-with-Intent-Detection/
│
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
│   ├── chatbot.py
│   └── app.py
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
    → Retrieve best matching answer
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

* Custom-built dataset (~100 Q&A pairs)
* Designed specifically for this project

#### Covered Domains

* Course and syllabus information
* Exams and grading policies
* Instructor and schedule queries
* NLP and AI concepts (e.g., transformers, embeddings)

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
* Best-matching question-answer pair is returned

This module performs semantic search over the knowledge base.

## Evaluation

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Retrieval Evaluation

* Top-1 Accuracy
* Qualitative analysis of chatbot responses
  
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

* Hybrid NLP system combining **intent classification** and **retrieval**
* Use of a real-world dataset (**CLINC150**) for intent modeling
* Integration of **transformer-based embeddings**
* Domain-adapted academic chatbot design
* Modular and extensible architecture

## Limitation

* Performance depends on knowledge base coverage

## Future Work

* Implement Retrieval-Augmented Generation (RAG)
* Replace classifier with fine-tuned BERT


