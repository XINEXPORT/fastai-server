import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_summary_with_memory(question, context_chunks, history):
    context = "\n\n".join(context_chunks)

    history_prompt = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-3:]])  # last 3 turns

    prompt = f"""
        You are a smart financial assistant. Use the context below and the previous conversation to answer the current question.

        Context:
        {context}

        Conversation so far:
        {history_prompt}

        User's current question:
        {question}

        Answer:
    """

    # Use openai.chat.completions.create instead of openai.ChatCompletion.create
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.5
    )
    # Access the content using the 'choices' attribute and then accessing the 'message' and 'content' attributes
    return response.choices[0].message.content
