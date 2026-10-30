# Buddy RAG Lab — Generative AI Assignment 2

An **independent** Streamlit RAG application inspired by [Buddy](https://github.com/anushkanaidu/buddy). The original repository is not modified.

## What it does

- Ingests **10 individually labeled, synthetic HR policy documents** for fictional Northstar Labs.
- Splits documents into **500-character chunks with 80-character overlap**, embeds using `sentence-transformers/all-MiniLM-L6-v2`, and indexes in **persistent ChromaDB**.
- Retrieves the top 3 chunks by cosine distance, shows source metadata, then asks **Groq Llama 3.1** to answer only from those chunks.
- Offers a **pure LLM no-retrieval** mode using the same model and generation temperature, plus a side-by-side comparison mode.
- Includes **10 evaluation questions**, document-level Precision@3/Recall@3 calculation, and a CSV export for answer comparisons.

> **Important:** These are constructed classroom documents, not real HR policies. Any accuracy claims must be based on actual observed outputs. No comparison results are fabricated.

## Local setup

Use Python 3.11 or 3.12, preferably in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your own GROQ_API_KEY to .env; never commit it
streamlit run app.py
```

First run downloads the embedding model, so it needs internet access. Indexing and retrieval run locally; text generation calls Groq. The UI will show an error if a key is missing. Run the automated evaluation from the repository root:

```bash
python evaluate.py                   # retrieve + score without a Groq key
python evaluate.py --compare         # also generates RAG and pure-LLM answers; requires a key
pytest -q
```

Inspect `results/evaluation.csv` after evaluation. For the assignment, manually score each answer as **correct / partially correct / incorrect / correctly abstained**, report exact observations, and include a short table of selected examples. `--compare` performs 20 calls total; API rate limits may apply.

## Architecture

`documents/*.md` → `RecursiveCharacterTextSplitter` → `all-MiniLM-L6-v2` embeddings → `Chroma PersistentClient` → top-3 chunks → Groq Llama 3.1 → Streamlit answer + evidence. In the baseline condition, the **same Groq model** receives only the question, no retrieved context. The UI retains Buddy's friendly HR assistant identity and Streamlit styling, but uses an independent codebase.

## Evaluation methodology

`tests/questions.csv` contains 10 questions (direct, list, cross-document, scenario, and unanswerable) with expected answers and relevant **documents**. Precision@3 counts the number of distinct relevant document sources among distinct source documents appearing in the top-3 chunks, divided by the number of distinct retrieved sources. Recall@3 divides distinct relevant sources retrieved by the number of labeled relevant source documents. **These are document-level proxies**, not strict chunk-level passage relevance: a retrieved chunk might come from the correct document but omit the answer. Inspect actual retrieved excerpts manually and mark irrelevant chunks; do not present document-level metrics as chunk-level precision. One unanswerable case has an empty gold set and is excluded from mean retrieval metrics but should be evaluated separately for abstention.

Potential threats to validity: synthetic corpus, short documents, manually assigned labels, model variability, question wording, and incomplete ground truth. Stronger evaluation would annotate every chunk independently and use more realistic document lengths, conflicting policy revisions and multiple human raters.

## Submission checklist

- [x] Independent repository files and functioning pipeline implementation
- [x] 10 documents, described chunk strategy, 10 questions, retrieval metrics script
- [x] RAG and pure LLM comparison implementation
- [ ] Run with dependency installation and real API key; capture actual outputs
- [ ] Review and export `results/evaluation.csv` with measured scores
- [ ] Complete `ANALYSIS_TEMPLATE.md` with **observed** results and insight
- [ ] Publish as its own GitHub repository, optionally deploy to Streamlit Community Cloud with `GROQ_API_KEY` configured as a secret

For reproducibility include Python version, dependencies, corpus, question set, retrieved chunks, settings, and any observed errors. Never upload `.env`, confidential employer documents, or generated `chroma_data/`.
