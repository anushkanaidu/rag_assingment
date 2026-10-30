# One-page analysis — Buddy RAG Lab

**Domain and purpose.** Buddy is a Q&A assistant for the fictional Northstar Labs HR policies. Ten synthetic documents cover onboarding, leave, verification, employee support and compliance. This application helps new hires locate rules without browsing separate files.

**Pipeline.** The system reads 10 Markdown files, splits each into overlapping 500-character chunks (80-character overlap), embeds them with all-MiniLM-L6-v2, and retrieves the three closest matches from ChromaDB. The Groq model receives this evidence and a grounding instruction. The comparison uses the same model with no retrieved passages.

**Experimental design.** I tested 10 questions across direct factual, multi-document, scenario, and unanswerable types. I labeled relevant source documents in advance. I calculated document-level Precision@3 and Recall@3 for questions with at least one relevant document and separately reviewed how the system handled the unanswerable case. These proxies do not establish passage-level relevance, so I also inspected retrieved excerpts.

**Measured results (fill in ONLY after running):** Mean document-level Precision@3: [VALUE]; Recall@3: [VALUE]. RAG fully correct: [X/10]. Pure LLM fully correct: [X/10]. Correct abstentions: RAG [X], pure LLM [X]. Include 2–3 actual question/answer examples from `results/evaluation.csv` in your submitted comparison table.

**Interpretation (complete using the real evidence):** On [QUESTION], RAG [ACTUAL BEHAVIOR] while the pure LLM [ACTUAL BEHAVIOR]. This difference matters because [WHY]. On [OTHER QUESTION], retrieved context [HELPED/FAILED] because [OBSERVED CHUNK].

**Limitations and improvements.** The policies are synthetic and short, and the same author created the questions and relevance labels. Document-level relevance can overestimate retrieval usefulness if a matched document chunk omits the answer. Next I would annotate individual chunks, add document revisions, test paraphrases/adversarial prompts, and report answer citation correctness and abstention separately. A larger realistic policy set would be needed before organizational deployment.

**Reflection:** [In your own words: what surprised you about context engineering and when pure prompting is sufficient versus retrieval?]
