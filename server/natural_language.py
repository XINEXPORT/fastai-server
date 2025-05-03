import nltk
from nltk.tokenize import sent_tokenize
from model_trainer import train_model_with_pdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json 
from tqdm import tqdm
import openai
import os 
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

nltk.download('punkt')
nltk.download('punkt_tab')

def search_chunks(model, query, embeddings, chunks, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def chunk_text(text, max_words=150):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    count = 0

    for sentence in sentences:
        chunk.append(sentence)
        count += len(sentence.split())

        if count >= max_words:
            chunks.append(" ".join(chunk))
            chunk = []
            count = 0

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


def get_openai_embeddings(texts, model="text-embedding-ada-002"):
    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

def main():
    financial_text = train_model_with_pdf("ltimindtree_annual_report.pdf")

    chunks = chunk_text(financial_text)
    print(f"✅ Total Chunks Created: {len(chunks)}")

 

    embeddings = get_openai_embeddings(chunks)
    

    data = [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]

    print(type(data))

    filename = "embeddings.json"


    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{filename} created.")
    else:
        print(f"{filename} already exists.")

if __name__ == "__main__":
    main()