
# NLP Practical Work (TPs)

This repository contains practical work notebooks for Natural Language Processing (NLP) course.

## Notebooks

### TP2: Introduction to Mathematical Optimization
**File:** `TP2_Introduction_maths_optimisation.ipynb`

Introduction to mathematical optimization concepts including:
- Analytical determination of critical points
- Function plotting and visualization
- Gradient and Hessian matrix calculations
- Contour line plotting
- Determining the nature of critical points

### TP3: Transformers Introduction
**File:** `TP3_Transformers_Introduction.ipynb`

Hands-on introduction to Hugging Face Transformers for various NLP tasks:
- Text Classification (sentiment analysis)
- Named Entity Recognition (NER)
- Question Answering
- Text Summarization
- Machine Translation
- Text Generation

### TP4: Transformer Text Classification
**File:** `TP4_Transformer_Text_Classification.ipynb`

Complete pipeline for emotion classification using Hugging Face transformers:
- Dataset exploration and handling class imbalance
- Tokenization with different models
- Feature extraction using pre-trained models
- Fine-tuning transformer models
- Model evaluation and error analysis
- Model deployment strategies

### TP5: Sequence-to-Sequence Tasks
**File:** `TP5_Sequence_to_Sequence_Tasks.ipynb`  
**🔗 [Open on Google Colab](https://colab.research.google.com/drive/1zHX-kwZJIJF-p4KGH66kuLaSAPkhoGGo?authuser=1)**

Exploring encoder-decoder architectures for machine translation and text summarization:
- Machine Translation using MarianMT, mBART, and T5
- Text Summarization with BART and T5
- Evaluation metrics (BLEU for translation, ROUGE for summarization)
- Handling long documents
- Fine-tuning templates

### TP6: Clustering and Topic Modeling
**File:** `TP6_Clustering_and_Topic_Modeling.ipynb`  
**🔗 [Open on Google Colab](https://colab.research.google.com/drive/12XBVKUTW-9j1IVXFGJSkr3kSTrQH4kUP?authuser=1#scrollTo=YFSNG30jxWGo)**

Clustering and topic modeling with BERTopic:
- Text clustering pipeline (embeddings, dimensionality reduction, clustering)
- Topic modeling with BERTopic
- Topic representation methods (KeyBERT, MMR, Flan-T5, Groq)
- Interactive visualizations
- Word clouds for topics

## Requirements

The notebooks require various Python packages. Each notebook includes installation instructions for the required dependencies.

Main packages used:
- `transformers` - Hugging Face Transformers library
- `datasets` - Hugging Face Datasets
- `torch` - PyTorch
- `sentence-transformers` - Sentence embeddings
- `bertopic` - Topic modeling
- `scikit-learn` - Machine learning utilities
- `matplotlib`, `plotly` - Visualization
- `umap-learn` - Dimensionality reduction
- `hdbscan` - Clustering

## Usage

Each notebook is self-contained and can be run independently. Follow the instructions within each notebook to install dependencies and execute the code.
