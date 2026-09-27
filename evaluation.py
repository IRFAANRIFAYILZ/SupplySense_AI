"""
evaluation.py
-------------
Evaluates four configurations on a fixed, hand-labeled test set
(llm/data/eval_test_set.json):

    A. Base LLM        -- no retrieval, no fine-tuning
    B. RAG + Base LLM   -- retrieval added
    C. QLoRA adapter    -- fine-tuned model, no retrieval
    D. RAG + QLoRA      -- both combined

Metrics computed:
    - Retrieval Recall@K   -- for retrieval-enabled configs (B, D): did the
                              retriever return a chunk from the expected
                              source document within the top K?
    - Groundedness         -- token-overlap between the generated answer and
                              the retrieved context it was given (a standard,
                              cheap proxy for faithfulness/hallucination used
                              when no LLM-judge is configured). If
                              ANTHROPIC_API_KEY is set, an LLM-judge scores
                              groundedness on a 0-1 scale instead (more
                              accurate) -- see `judge_groundedness_llm`.
    - Answer relevance     -- TF-IDF cosine similarity between the question
                              and the generated answer (proxy for "did the
                              answer actually address the question").

HONESTY NOTE (read this before trusting any number this script prints):
    - Retrieval Recall@K is ALWAYS genuinely computed here, using the real
      TF-IDF retriever built by rag/ingestion.py. No fabrication.
    - Configs A (Base LLM) and C/D (QLoRA) require an actual LLM to generate
      answers. In an environment with no GPU and no internet access (like
      this project's authoring sandbox), those generation calls cannot be
      made. Rather than inventing plausible-looking scores, this script
      marks those rows "NOT RUN -- requires GPU/API key" and explains what
      to set up to actually run them (see README).
    - Config B (RAG) DOES run everywhere, because rag/qa.py has a built-in
      extractive fallback (returns real document text, no LLM required) --
      so its groundedness score is genuinely computed, not fabricated (and
      will legitimately show a high groundedness score, since an extractive
      answer trivially IS the context).

Run:
    python llm/evaluation.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np

from rag.retrieval import Retriever
from rag.qa import answer_question, NOT_ENOUGH_INFO

BASE_DIR = Path(__file__).parent
TEST_SET_PATH = BASE_DIR / "data" / "eval_test_set.json"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TOP_K = 3


def load_test_set():
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def word_set(text):
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def token_overlap_groundedness(answer: str, context: str) -> float:
    """Fraction of the answer's content words that also appear in the
    retrieved context. 1.0 = every word in the answer is supported by the
    context (as with an extractive answer); lower values suggest the answer
    may contain unsupported content."""
    a_words = word_set(answer)
    c_words = word_set(context)
    if not a_words:
        return 0.0
    overlap = a_words & c_words
    return round(len(overlap) / len(a_words), 3)


def tfidf_relevance(question: str, answer: str) -> float:
    """Cosine similarity between question and answer using a lightweight
    TF-IDF vectorizer fit just on this pair -- a real, computable proxy for
    'does the answer address the question' that needs no external model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not answer.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit([question, answer])
    q_v, a_v = vec.transform([question, answer]).toarray()
    denom = (np.linalg.norm(q_v) * np.linalg.norm(a_v))
    if denom == 0:
        return 0.0
    return round(float(np.dot(q_v, a_v) / denom), 3)


def judge_groundedness_llm(question, answer, context):
    """Optional, more accurate groundedness scorer using Claude as an
    LLM-judge. Only used if ANTHROPIC_API_KEY is set; otherwise callers
    should fall back to token_overlap_groundedness."""
    import anthropic
    client = anthropic.Anthropic()
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "On a scale of 0.0 to 1.0, how well is the Answer supported ONLY by "
        "the Context (1.0 = fully supported, 0.0 = not supported / "
        "hallucinated)? Respond with ONLY the number."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return float(msg.content[0].text.strip())
    except ValueError:
        return None


def evaluate_retrieval(retriever, test_set):
    hits = 0
    scored = 0
    per_question = []
    for item in test_set:
        if not item["answerable"]:
            continue
        results = retriever.retrieve(item["question"], top_k=TOP_K)
        retrieved_docs = {r["doc"] for r in results}
        hit = item["expected_doc"] in retrieved_docs
        hits += int(hit)
        scored += 1
        per_question.append({
            "question": item["question"], "expected_doc": item["expected_doc"],
            "retrieved_docs": list(retrieved_docs), "hit": hit,
        })
    recall_at_k = round(hits / scored, 3) if scored else None
    return recall_at_k, per_question


def evaluate_rag_config(retriever, test_set):
    """Config B: RAG + (Base LLM if API key set, else extractive fallback).
    This ALWAYS runs, and every number produced here is genuinely computed."""
    groundedness_scores = []
    relevance_scores = []
    refusal_correct = 0
    refusal_total = 0
    use_llm_judge = bool(os.environ.get("ANTHROPIC_API_KEY"))

    for item in test_set:
        result = answer_question(item["question"], retriever=retriever, top_k=TOP_K)
        answer = result["answer"]

        if not item["answerable"]:
            refusal_total += 1
            if answer.strip() == NOT_ENOUGH_INFO:
                refusal_correct += 1
            continue

        # Reconstruct context for groundedness scoring
        chunks = retriever.retrieve(item["question"], top_k=TOP_K)
        context = "\n".join(c["text"] for c in chunks)
        if not context:
            continue

        if use_llm_judge:
            g = judge_groundedness_llm(item["question"], answer, context)
            if g is None:
                g = token_overlap_groundedness(answer, context)
        else:
            g = token_overlap_groundedness(answer, context)

        groundedness_scores.append(g)
        relevance_scores.append(tfidf_relevance(item["question"], answer))

    refusal_accuracy = round(refusal_correct / refusal_total, 3) if refusal_total else None

    return {
        "groundedness": round(float(np.mean(groundedness_scores)), 3) if groundedness_scores else None,
        "answer_relevance": round(float(np.mean(relevance_scores)), 3) if relevance_scores else None,
        "correct_refusal_rate": refusal_accuracy,
        "generation_backend": "anthropic (LLM-judge groundedness)" if use_llm_judge else "extractive (no API key set)",
    }


def main():
    test_set = load_test_set()
    retriever = Retriever()

    print("=== Retrieval Evaluation (genuinely computed) ===")
    recall_at_k, per_question = evaluate_retrieval(retriever, test_set)
    print(f"Recall@{TOP_K}: {recall_at_k}")
    for pq in per_question:
        status = "HIT " if pq["hit"] else "MISS"
        print(f"  [{status}] '{pq['question'][:50]}...' expected={pq['expected_doc']} "
              f"retrieved={pq['retrieved_docs']}")

    print("\n=== Config B: RAG (+ Base LLM or extractive fallback) -- genuinely computed ===")
    rag_results = evaluate_rag_config(retriever, test_set)
    for k, v in rag_results.items():
        print(f"  {k}: {v}")

    print("\n=== Config A: Base LLM (no retrieval) ===")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        print("  An API key is set, but Config A specifically measures the model WITHOUT "
              "retrieval context; wire this up by calling the same LLM with no context if "
              "you want this comparison row -- left as an exercise since the project's "
              "focus is the RAG/LoRA comparison, not raw base-LLM behavior.")
    else:
        print("  NOT RUN -- no ANTHROPIC_API_KEY / OPENAI_API_KEY set, and no local model "
              "loaded. Set an API key (see .env.example) to enable this comparison row.")

    print("\n=== Configs C & D: QLoRA adapter (with/without RAG) ===")
    adapter_dir = BASE_DIR / "artifacts" / "lora_adapter"
    if adapter_dir.exists():
        print("  Adapter found. Run llm/inference.py --adapter on a GPU machine to generate "
              "answers for this config, then re-run this script's groundedness/relevance "
              "scoring against those outputs.")
    else:
        print("  NOT RUN -- no LoRA adapter found at llm/artifacts/lora_adapter. "
              "Train one first: python llm/lora_training.py (requires a GPU; see README).")

    summary = {
        "retrieval_recall_at_k": recall_at_k,
        "top_k": TOP_K,
        "configs": {
            "A_base_llm": "NOT RUN (see console output / README)",
            "B_rag": rag_results,
            "C_qlora": "NOT RUN (requires trained adapter + GPU inference; see README)",
            "D_rag_qlora": "NOT RUN (requires trained adapter + GPU inference; see README)",
        },
        "per_question_retrieval": per_question,
    }
    with open(ARTIFACT_DIR / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full evaluation report to {ARTIFACT_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
