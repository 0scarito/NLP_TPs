# NLP Labs

Graduate NLP coursework (CentraleSupélec × ESSEC, AIDAMS program). Five Jupyter notebooks that go from optimization fundamentals to a full Hugging Face fine-tuning pipeline, a six-method text-classification benchmark, and topic modeling of ~45K ArXiv abstracts with BERTopic.

## Contents

| Notebook | What it covers |
|---|---|
| [`TP2_Introduction_maths_optimisation.ipynb`](TP2_Introduction_maths_optimisation.ipynb) | Optimization foundations for ML: analytical critical points, gradients and Hessian matrices, eigenvalue-based classification of stationary points, 3D surface and contour plots (NumPy/Matplotlib). |
| [`TP3_Transformers_Introduction.ipynb`](TP3_Transformers_Introduction.ipynb) | Hugging Face `pipeline` API across six core NLP tasks: text classification, NER, question answering, summarization, translation (Helsinki-NLP MarianMT), and text generation. |
| [`TP4_Transformer_Text_Classification.ipynb`](TP4_Transformer_Text_Classification.ipynb) | End-to-end emotion classification on the `emotion` dataset: class-imbalance handling, tokenizer comparison (DistilBERT vs BERT vs RoBERTa), feature extraction with frozen encoders + UMAP visualization, fine-tuning DistilBERT/RoBERTa/ALBERT with the `Trainer` API, error analysis, and deployment patterns (HF Inference API, Flask). |
| [`TP5/TP5.ipynb`](TP5/TP5.ipynb) | Six approaches to sentiment classification benchmarked on the same Rotten Tomatoes test set — task-specific model, embeddings + classifier, prototype similarity, zero-shot, LLM prompting, and text-to-text generation (results below). |
| [`TP6/TP6.ipynb`](TP6/TP6.ipynb) | Clustering and topic modeling of ~45K ArXiv NLP paper abstracts: sentence embeddings → UMAP → HDBSCAN (155 clusters), then BERTopic with multiple topic-representation models, including LLM-generated topic labels. |

## Key techniques

**Transformer fine-tuning and evaluation (TP4)**
- Fine-tuning `distilbert-base-uncased`, `roberta-base`, and `albert-base-v2` for 6-class emotion classification with the Hugging Face `Trainer` API.
- Feature-extraction baseline: frozen DistilBERT hidden states fed to scikit-learn classifiers — logistic regression reaches 0.63 accuracy vs a 0.35 dummy baseline on validation embeddings.
- Class-imbalance strategies (class weights, `imbalanced-learn` over/under-sampling), confusion-matrix error analysis, UMAP projection of embeddings.

**Classification method benchmark (TP5)** — accuracy on 1,066 Rotten Tomatoes test reviews:

| Method | Model | Accuracy |
|---|---|---|
| Pre-trained task-specific model | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 0.80 |
| Embeddings + logistic regression | `all-mpnet-base-v2` + scikit-learn | 0.85 |
| Unsupervised prototype similarity | class-mean embeddings, cosine similarity | 0.84 |
| Zero-shot via label embeddings | embed label descriptions, cosine similarity | 0.78 |
| LLM prompting | Llama 4 Scout via Groq API | sampled only |
| Text-to-text generation | `google/flan-t5-small` | 0.84 |

**Topic modeling at scale (TP6)**
- Classic pipeline built by hand: `gte-small` sentence embeddings (384-d) → UMAP to 5 dimensions → HDBSCAN (`min_cluster_size=50`, 155 clusters found).
- BERTopic with c-TF-IDF topic extraction, compared against richer representation models: KeyBERTInspired, Maximal Marginal Relevance, Flan-T5 text generation, and a custom Groq-powered function that relabels every topic with an LLM (with retry/rate-limit handling).
- Interactive cluster visualizations and per-topic word clouds.

**Foundations (TP2, TP3)**
- Analytical optimization: gradients, Hessians, eigenvalue tests for the nature of critical points — the math behind gradient-based training.
- Survey of the main NLP task families through Hugging Face pipelines before building anything custom.

## Running

Notebooks are self-contained (each starts with its own `pip install` cell) and were developed on Google Colab — a GPU runtime is recommended for TP4–TP6.

Local setup:

```bash
pip install transformers datasets torch accelerate sentence-transformers \
    scikit-learn imbalanced-learn umap-learn hdbscan bertopic \
    matplotlib plotly wordcloud groq
jupyter lab
```

The LLM sections of TP5 and TP6 call the Groq API — set your own key via the `GROQ_API_KEY` environment variable.
