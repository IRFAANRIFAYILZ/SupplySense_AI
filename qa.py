"""
qa.py
-----
RAG Step 6-7: Take retrieved chunks + the user's question, and generate a
grounded answer that always cites its sources.

Generation backend (checked in this priority order):
  1. ANTHROPIC_API_KEY set in environment -> calls Claude (recommended)
  2. OPENAI_API_KEY set in environment -> calls OpenAI's API
  3. Neither key set -> EXTRACTIVE fallback: no external LLM call is made.
     The system returns the single highest-scoring retrieved chunk verbatim
     as the answer, clearly labeled as an extractive (non-generative) answer.
     This is NOT a hallucination risk (it literally quotes the source
     document) and keeps the whole RAG pipeline runnable offline / without
     any API key, which matters for a student project with no billing setup.

In all cases: if retrieval finds no sufficiently relevant chunks, the system
returns the fixed, honest response required by the spec:
  "The available documents do not contain enough information to answer this."
and makes NO generation call at all (avoids hallucination by construction).
"""

import os
from pathlib import Path

from rag.retrieval import Retriever

NOT_ENOUGH_INFO = "The available documents do not contain enough information to answer this."

SYSTEM_PROMPT = """You are a policy assistant for a fictional company, SupplySense
Manufacturing Group. Answer the user's question using ONLY the provided document
excerpts. Do not use outside knowledge. Do not invent policy details that are not
present in the excerpts. If the excerpts do not actually answer the question,
respond with exactly: "{not_enough}" Keep answers concise (2-4 sentences).""".format(
    not_enough=NOT_ENOUGH_INFO
)


def _format_context(chunks):
    return "\n\n".join(
        f"[Source: {c['doc']} — {c['section']}]\n{c['text']}" for c in chunks
    )


def _generate_with_anthropic(question, chunks):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    context = _format_context(chunks)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
        }],
    )
    return message.content[0].text.strip()


def _generate_with_openai(question, chunks):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    context = _format_context(chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def _generate_extractive(question, chunks):
    """No-API-key fallback: return the top chunk's text directly, labeled as
    extractive. This guarantees the answer is 100% grounded (it's a literal
    excerpt) at the cost of not being a fluent generated sentence."""
    top = chunks[0]
    return (
        f"[Extractive answer — no LLM API key configured, showing the most "
        f"relevant excerpt directly]\n\n{top['text']}"
    )


def answer_question(question: str, retriever: Retriever = None, top_k: int = 3):
    if retriever is None:
        retriever = Retriever()

    chunks = retriever.retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": NOT_ENOUGH_INFO,
            "sources": [],
            "mode": "no_retrieval",
        }

    if os.environ.get("ANTHROPIC_API_KEY"):
        mode = "anthropic"
        try:
            answer = _generate_with_anthropic(question, chunks)
        except Exception as e:
            answer = f"(Generation error, falling back to extractive mode: {e})\n\n" + \
                     _generate_extractive(question, chunks)
            mode = "extractive_fallback_error"
    elif os.environ.get("OPENAI_API_KEY"):
        mode = "openai"
        try:
            answer = _generate_with_openai(question, chunks)
        except Exception as e:
            answer = f"(Generation error, falling back to extractive mode: {e})\n\n" + \
                     _generate_extractive(question, chunks)
            mode = "extractive_fallback_error"
    else:
        mode = "extractive"
        answer = _generate_extractive(question, chunks)

    return {
        "answer": answer,
        "sources": [f"{c['doc']} — {c['section']} (score: {c['score']})" for c in chunks],
        "mode": mode,
    }


if __name__ == "__main__":
    test_questions = [
        "What happens if a supplier repeatedly misses the delivery deadline?",
        "What is the penalty for late delivery?",
        "What are the requirements for supplier evaluation?",
        "What is the minimum inventory level?",
        "What is the best pizza topping?",
    ]
    retriever = Retriever()
    for q in test_questions:
        result = answer_question(q, retriever=retriever)
        print(f"\nQ: {q}")
        print(f"Mode: {result['mode']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
