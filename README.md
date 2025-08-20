## Patent Topic Modeling & Similarity Search

This project analyzes a set of patents to automatically discover thematic groupings and enable semantic similarity search, using patent data from the [HUPD dataset](https://huggingface.co/datasets/HUPD/hupd). For this demo, a subset of patents that **received a patent number**, and filed between **2013 and 2017** was used.

**🔗 Live Demo:** [View on Streamlit](https://patents-topic-modeling.streamlit.app)
The Streamlit app provides a sample of high-level statistics on the topic results, including insights into top topics, topics over the years, and selected topics that experienced significant growth.

> ⏳ *Note: If the web app has been idle (“asleep”), it may take 1–2 minutes to load. You may need to actively wake it up by clicking “Yes, get this app back up!” before it becomes responsive.*


---

### 1. Topic Modeling with BERTopic

Each patent’s text (title, abstract) is converted into dense semantic embeddings using a pretrained [SentenceTransformer](https://www.sbert.net/) model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). These embeddings capture contextual meaning, allowing us to go beyond keyword matching. A [BERTopic](https://maartengr.github.io/BERTopic/) model is then trained on these embeddings to cluster patents into coherent topics. Each patent is assigned to its most relevant topic, producing a high-level thematic map of the corpus.

**Runtimes (approximate, GPU-based, full HUPD subset - 2013–2017):**
- Loading the dataset ~ 1 hour
- Creating embeddings ~ 1 hour
- Training BERTopic on the embeddings ~1 hour

**Side note # 1 - Train/val datasets**
Train/val splits exist in code but are not for ML training; they exist only for compatibility with Hugging Face's `load_dataset` API and the HUPD dataset loader. The entire dataset (2013-2017) is used for BERTopic training in this notebook.

**Side note # 2 – why train BERTopic rather than using a pretrained model?**
Training BERTopic on the 2013–2017 HUPD dataset ensures that the discovered topics reflect the patents’ vocabulary, technical fields, and filing patterns. A pretrained topic model, trained on general text, would likely miss domain-specific nuances, misclassify patents, and produce irrelevant topics.

### 2. Similarity Search with FAISS

To enable fast retrieval of related patents, the embeddings are indexed using [FAISS](https://faiss.ai/), a high-performance vector search library. This allows quick nearest-neighbor lookups, meaning for any given patent, we can quickly retrieve likely similar patents.
Patent Explorer in Streamlit app: given a specific patent, finds the top k most similar patents. Current version works on sample dataset only.

---

This combination provides both **topic-level understanding** of the patent landscape and **fine-grained similarity search** capabilities.

---

## Setup & Usage Instructions

### 1. Environment Installation

**Python version:** 3.11.13

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   [Guide: Python Virtual Environments](https://docs.python.org/3/library/venv.html)

2. **Activate the environment**:

   * **Windows:** `venv\Scripts\activate`
   * **Mac/Linux:** `source venv/bin/activate`

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

### 2. Environment Variables

The project requires a `.env` file in the root directory to store API tokens and configuration:

```ini
# Hugging Face
HF_TOKEN=""
HF_USERNAME=""

# Dataset
DATASET_TYPE=""  # "SAMPLE" or "FULL"
UPLOAD_TO_AWS_S3="" # "True" or "False"

# AWS
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_DEFAULT_REGION=""
S3_BUCKET=""
```

> Make sure to fill in your Hugging Face credentials and AWS S3 access details. `DATASET_TYPE` controls whether to use a sample subset or the full dataset. The AWS credentials are used for both reading from and writing to S3, so ensure the credentials have the necessary permissions.

---

### 3. Running the Model & Generating Outputs

* **Train the topic model and generate outputs:**
   Open jupyterlab
   ```bash
   jupyter lab
   ```

  Run `patents_topic_modeling.ipynb`

* **Explore trending topics over time:**
   Open jupyterlab
   ```bash
   jupyter lab
   ```
   Run `trending_topics.ipynb`

---

### 4. Running the Streamlit Dashboard

Launch the interactive app with:

```bash
streamlit run patents_dashboard.py
```

The app allows you to:

* View high-level statistics on:

  * Selected topics that experienced growth
  * Patent topics distribution
  * Patents received a patent number, filed over the years

* Find similar patents *(work-in-progress feature)*

**Note on Reproducibility:**
BERTopic was trained without a fixed random seed. As a result, topic assignments, ordering, and counts may vary between runs. The dashboard reflects one particular run and is for illustrative purposes only.


### 5. Repository Structure

```
├─ README.md                      # Project overview and setup instructions
├─ LICENSE                        # MIT License
├─ patents_topic_modeling.ipynb   # Notebook to train BERTopic on patent data
├─ trending_topics.ipynb          # Notebook for analyzing trending topics
├─ s3_utils.py                    # Utility functions for S3 data handling
├─ hupd_modified.py               # HUPD dataset loader, adapted from official repo
├─ requirements.txt               # Python dependencies
├─ patents_dashboard.py           # Main Streamlit app for visualization & exploration
├─ pages/
│  └─ patent_explorer.py          # Streamlit page for exploring similar patents
└─ .env                           # Environment variables (API keys, dataset type, AWS)
```

- Note: hupd_modified.py script is a slight modification optimized for local disk storage compared to the original script [hupd.py](https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd.py)


### 6. Further Research 

1. **Hybrid Topic Labeling**

   * **Problem:** Current BERTopic topics are based solely on text embeddings (title + abstract) and do not incorporate structured, domain-specific classifications such as CPC (Cooperative Patent Classification) or IPC (International Patent Classification), both of which are available in the HUPD dataset.  
   * **Solution:** Combine unsupervised topics from BERTopic with CPC/IPC labels:  
     * Use BERTopic to discover semantic clusters.  
     * Map these clusters to CPC/IPC labels by combining semantic similarity between a topic and CPC/IPC label descriptions with majority voting over the patents’ existing labels in the cluster.  
     * This creates a hybrid labeling system that captures both latent semantic patterns and structured domain categories.  
   * **Benefit:** Topics reflect both the language used in patents and the formal classification system, improving interpretability and relevance. Each topic can map to multiple CPC/IPC labels, enabling more precise mapping between semantic clusters and domain classifications.

2. **Hierarchical Topic Modeling**

   * **Problem:** Some patent topics are broad, while others are highly specialized, which flat clustering approaches may not fully capture.
   * **Solution:** Implement a hierarchical topic model:
     * Level 1: Broad technical areas (e.g., "Computer Hardware").
     * Level 2: More specific subtopics (e.g., "GPU Architecture" or "Memory Optimization").
   * **Benefit:** The resulting model supports exploration at multiple levels of granularity, from high-level surveys to detailed technical analysis.

3. **Multi-Label Topic Modeling**

   * **Problem:** Patents often cover multiple technical areas. Assigning each patent to a single topic may oversimplify the innovation landscape.
   * **Solution:** Use multi-label models or thresholded topic probabilities from BERTopic:
     * Assign multiple topics to a patent if its embedding aligns with several clusters.
     * Leverage probabilistic outputs from BERTopic to identify secondary and tertiary topics.
   * **Benefit:** Reflects the true multi-disciplinary nature of patents, enabling richer analyses of co-occurring technologies and cross-domain innovation patterns.