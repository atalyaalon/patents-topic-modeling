## Patent Topic Modeling & Similarity Search

This project analyzes a set of patents to automatically discover thematic groupings and enable semantic similarity search, using patent data from the [HUPD dataset](https://huggingface.co/datasets/HUPD/hupd). For this demo, a subset of patents that **received a patent number**, and filed between **2013 and 2017** was used.

**🔗 Live Demo:** [View on Streamlit](https://patents-topic-modeling.streamlit.app)
The Streamlit app provides a sample of high-level statistics on the topic results, including insights into top topics, topics over the years, and selected topics that experienced significant growth.

---

### 1. Topic Modeling with BERTopic

Each patent’s text (title, abstract) is converted into dense semantic embeddings using a pretrained [SentenceTransformer](https://www.sbert.net/) model. These embeddings capture contextual meaning, allowing us to go beyond keyword matching. A [BERTopic](https://maartengr.github.io/BERTopic/) model is then trained on these embeddings to cluster patents into coherent topics. Each patent is assigned to its most relevant topic, producing a high-level thematic map of the corpus.

### 2. Similarity Search with FAISS

To enable fast retrieval of related patents, the embeddings are indexed using [FAISS](https://faiss.ai/), a high-performance vector search library. This allows quick nearest-neighbor lookups, meaning for any given patent, we can instantly find others with similar content.
WIP: Patent Explorer in Streamlit app: given a specific patent, finds the the top k most similar patents

---

This combination provides both **topic-level understanding** of the patent landscape and **fine-grained similarity search** capabilities.

---

## Setup & Usage Instructions

### 1. Environment Installation

**Python version:** 3.11.3

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
DATASET_TYPE=""  # "SAMPLE" or "FULL"

# AWS
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_DEFAULT_REGION=""
S3_BUCKET=""
```

> Make sure to fill in your Hugging Face credentials and AWS S3 access details. `DATASET_TYPE` controls whether to use a sample subset or the full dataset.

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

---

### 5. Repository Structure

```
├─ README.md                      # Project overview and setup instructions
├─ LICENSE                        # MIT License
├─ patents_topic_modeling.ipynb   # Notebook to train BERTopic on patent data
├─ trending_topics.ipynb          # Notebook for analyzing trending topics
├─ s3_utils.py                    # Utility functions for S3 data handling
├─ hupd_modified.py               # HUPD dataset loader (optimized for disk storage), adapted from official repo
├─ requirements.txt               # Python dependencies
├─ patents_dashboard.py           # Main Streamlit app for visualization & exploration
├─ pages/
│  └─ patent_explorer.py          # Streamlit page for exploring similar patents
└─ .env                           # Environment variables (API keys, dataset type, AWS)
```