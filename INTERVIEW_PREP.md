# Interview Preparation — SupplySense AI

Straightforward answers you should be able to give confidently, in your own
words, without notes.

---

**1. Why did you choose this problem?**
Supply chain management is a domain where structured data (orders,
inventory), unstructured data (contracts, policies), and decision-support
needs all coexist naturally — which let me demonstrate ML, RAG, and
fine-tuning in one coherent product instead of three disconnected demos.

**2. Why RAG instead of just putting the policies in the prompt?**
The documents are small here, but in a real company, policy documents are
large, change often, and shouldn't require re-training a model every time
they're updated. RAG retrieves only the relevant chunks at query time, so
the system always answers from the current version of the documents and
never needs retraining just because a policy changed.

**3. How does your RAG pipeline work, step by step?**
Load the markdown documents → split them into chunks along section headers
→ embed each chunk into a vector → store the vectors → at query time, embed
the question the same way → find the chunks with the highest cosine
similarity → pass only those chunks to the LLM (or return them directly if
no LLM is configured) → the answer always lists which document/section it
came from.

**4. Why use embeddings instead of exact keyword search?**
Embeddings capture semantic similarity, so a question phrased differently
from the document text can still find the right passage. My implementation
actually falls back to TF-IDF (a lexical vector method) when a neural
embedding model isn't available, which is a good, honest illustration of the
trade-off: TF-IDF is fast and needs no GPU, but it only matches on shared
vocabulary, while neural embeddings generalize across paraphrasing.

**5. Why FAISS/Chroma? Did you use one?**
For six documents split into ~46 chunks, a full similarity scan over a NumPy
array is fast enough that a vector database isn't necessary — I store
embeddings as a `.npy` file instead. FAISS or Chroma earns its keep once
you're indexing thousands to millions of chunks and need approximate
nearest-neighbor search to stay fast; I designed the code so swapping one in
later doesn't require restructuring the pipeline.

**6. What is LoRA?**
Low-Rank Adaptation. Instead of updating all of a model's weights during
fine-tuning, you freeze the base model and inject small, trainable low-rank
matrices into specific layers (here, the attention projections). You train
only those small matrices, which is dramatically cheaper in memory and
compute while still meaningfully changing the model's behavior.

**7. What is QLoRA?**
QLoRA adds one more trick: the frozen base model is loaded in 4-bit
quantized precision (NF4) instead of full precision, cutting memory use
further, while the LoRA adapters themselves are still trained in higher
precision. That's what makes fine-tuning a model feasible on a single
consumer GPU or a free Colab instance.

**8. Why QLoRA instead of full fine-tuning?**
Full fine-tuning of even a small model requires storing gradients and
optimizer states for every parameter, which needs far more GPU memory than
a student typically has access to. QLoRA gets you a genuinely fine-tuned
model on a fraction of the hardware, at some cost in flexibility compared to
updating every weight.

**9. What exactly did you fine-tune, and on what data?**
`Qwen2.5-0.5B-Instruct`, using 240 instruction examples I generated
programmatically from the actual synthetic supplier/order data — supplier
summaries, risk explanations, recommendations, and trend descriptions,
where the target text was built from real computed numbers (on-time rates,
delay counts, risk scores), not hand-written or invented. The goal wasn't
to teach the model new facts (that's RAG's job) but to teach it to respond
in SupplySense's structured business format consistently.

**10. How did you evaluate the model(s)?**
A fixed, hand-labeled test set of 13 questions (10 answerable from the
documents, 3 not). I measure Retrieval Recall@K (did the retriever find a
chunk from the correct source document), groundedness (how much of the
generated answer is actually supported by the retrieved context), answer
relevance (does the answer address the question), and refusal accuracy (did
the system correctly say "not enough information" on the 3 unanswerable
questions). I was careful to only report numbers I could actually compute —
the Base-LLM and QLoRA rows of my comparison table are explicitly marked
"not run" rather than filled with plausible-looking scores, because I
didn't have GPU/API access in the environment where I built this.

**11. How does the ML model work?**
A RandomForestClassifier predicts whether a new order will be delayed,
using historical supplier features (on-time rate, average delivery days,
quality score, risk score) plus order-specific features (quantity, month,
category, priority). Random forests handle mixed numeric/categorical
features well, are relatively robust to overfitting with enough trees, and
give you feature importances for free, which matters for explainability.

**12. How does anomaly detection work?**
Two methods: a Z-score check comparing each supplier's current-month
average delay to their own historical average and standard deviation
(flagging sudden, statistically unusual worsening), and an Isolation Forest
run independently over the same supplier-month data as a cross-check using
a completely different algorithm. Isolation Forest works by randomly
partitioning the data — anomalies get isolated in fewer splits than normal
points, because they're "different" along more dimensions.

**13. How did you generate your dataset?**
Programmatically, with a fixed random seed for reproducibility. I
deliberately built in structure rather than pure randomness: a subset of
suppliers are seeded as chronically underperforming, three suppliers get an
injected sudden delay spike in a specific month (to give the anomaly
detector something real to find), and delay probability increases in the
Nov–Jan peak season. Supplier-level aggregate stats (on-time rate, average
delivery days) are then recomputed from the actual generated orders, so
everything is internally consistent rather than independently sampled.

**14. How do you prevent hallucination?**
Two layers: first, retrieval has a minimum similarity-score cutoff — if
nothing in the documents is actually relevant, the system returns a fixed
"not enough information" message instead of asking the LLM to answer
anyway. Second, when no LLM API key is configured, the system uses an
extractive fallback that returns the literal retrieved text rather than a
generated paraphrase, which by construction cannot hallucinate.

**15. How does the system handle missing information?**
By design, it says so explicitly rather than guessing — both in RAG (fixed
refusal message when retrieval confidence is too low) and in the NL data
analytics router (an honest "I couldn't confidently map this question"
response when no known intent matches).

**16. How does natural-language data analysis work, and why no `eval()`?**
A question is matched against a small set of keyword/regex patterns to pick
one of a handful of pre-written, tested pandas functions (e.g. "most delayed
supplier," "suppliers above a delay threshold"). The system never generates
or executes new code — it only selects which vetted function to call and
extracts simple parameters (a number, a name) from the question. This is a
hard requirement for safety: letting an LLM write and run arbitrary Python
against your data is a real security and correctness risk in production.

**17. What are the limitations of this project?**
The TF-IDF retrieval fallback is lexical, so it can miss paraphrased
questions that share no vocabulary with the source text. The delay
prediction model's precision/recall (~0.29/0.51) reflect a genuinely noisy
problem rather than a highly accurate classifier. The NL analytics router
only covers six intents. And the LoRA/QLoRA training and full four-way
model comparison require GPU hardware I didn't have while building this, so
those pieces are complete and correct code that I haven't personally run
end-to-end.

**18. What would you improve in a production version?**
Swap TF-IDF for a proper neural embedding model with a real vector database
(FAISS/Chroma/pgvector) once semantic recall matters at scale; add real
time-series inventory snapshots instead of a current-state-only check;
expand the NL analytics intents or add a constrained LLM-based intent
classifier in front of the same safe function registry; and run the full
model evaluation with the LoRA adapter trained and an LLM judge configured,
rather than leaving those rows marked "not run."

**19. Why a RandomForest instead of a neural network for delay prediction?**
The dataset is tabular with under 4,000 rows and a handful of features —
exactly the regime where gradient-boosted trees or random forests typically
outperform or match neural networks, train in seconds, don't need GPU, and
give interpretable feature importances. A neural network would add
complexity without a clear accuracy benefit here.

**20. How would you know if this system was working well in production?**
I'd track whether high-risk-flagged suppliers actually correlate with real
future delays (precision/recall on live outcomes, not just the held-out
test set), whether RAG answers are being marked "not enough information"
too often (a sign documents need updating or retrieval needs tuning), and
whether the anomaly detector's false-positive rate is tolerable for the
people acting on its alerts.

**21. What's the difference between what RAG gives you and what LoRA gives you, concretely?**
If a company updates its late-delivery penalty from 0.5% to 1% per day, RAG
picks that up immediately the next time someone edits the document — no
retraining. LoRA wouldn't know about that change until you fine-tuned again
on new examples. Conversely, if you want the model to *always* format its
answer as "Supplier Risk: X / Possible actions: 1... 2... 3...", RAG alone
won't reliably enforce that formatting — that's a style/behavior change,
which is what LoRA is suited for.
