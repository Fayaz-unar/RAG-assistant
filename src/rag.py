from groq import Groq

from .embeddings import create_embeddings
from .vector_store import search_vector_store
from .prompts import SYSTEM_PROMPT, create_rag_prompt


def retrieve_context(
    question,
    index,
    chunks,
    top_k=5
):
    query_embedding = create_embeddings([question])[0]

    scores, indices = search_vector_store(
        index,
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for score, index_number in zip(scores, indices):

        if index_number == -1:
            continue

        chunk = chunks[index_number].copy()
        chunk["score"] = float(score)

        retrieved_chunks.append(chunk)

    return retrieved_chunks


def generate_answer(
    client,
    question,
    retrieved_chunks
):
    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
Document: {chunk['document']}
Page: {chunk['page']}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = create_rag_prompt(
        question,
        context
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="medium"
    )

    return response.choices[0].message.content
