import os
import json
import numpy as np
import openai
import faiss
from tqdm import tqdm
import PyPDF2
import tweepy
from datetime import datetime

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Trending Topics (Hardcoded) ---
def fetch_trending_topics_hardcoded(num_topics=10):
    sample_trending_topics = [
        "AI advancements", "SpaceX Mars mission", "Climate change action", "Olympics 2025",
        "Electric vehicles", "Quantum computing", "Remote work trends", "Crypto regulations",
        "Healthy living", "Augmented reality", "Generative AI tools", "Renewable energy",
        "Fintech innovation", "Digital marketing", "Personal branding", "Sustainable startups",
        "5G networks", "Wearable tech", "Blockchain adoption", "Tech IPOs"
    ]
    return sample_trending_topics[:num_topics]

# --- File Extraction ---
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        text = f"Error reading PDF: {e}"
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading TXT: {e}"

# --- User Settings/File Handling ---
def save_user_settings(user_id, settings: dict):
    with open(os.path.join(DATA_DIR, f"{user_id}_settings.json"), "w") as f:
        json.dump(settings, f)

def save_uploaded_file(user_id, file):
    file_path = os.path.join(DATA_DIR, f"{user_id}_{file.name}")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_user_settings(user_id):
    settings_path = os.path.join(DATA_DIR, f"{user_id}_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            return json.load(f)
    return None

# --- RAG: Chunk, Embed, Save/Load, Retrieve ---
def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_openai_embedding(text, api_key=None):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def embed_chunks(chunks, api_key=None):
    embeddings = []
    for chunk in tqdm(chunks):
        emb = get_openai_embedding(chunk, api_key)
        embeddings.append(emb)
    return np.vstack(embeddings)

def save_embeddings(user_id, embeddings, chunks):
    np.save(f"data/{user_id}_embeddings.npy", embeddings)
    with open(f"data/{user_id}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_embeddings(user_id):
    emb_path = f"data/{user_id}_embeddings.npy"
    chunk_path = f"data/{user_id}_chunks.json"
    if not (os.path.exists(emb_path) and os.path.exists(chunk_path)):
        return None, None
    embeddings = np.load(emb_path)
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks

def retrieve_relevant_chunks(user_id, query, top_k=1, api_key=None):
    embeddings, chunks = load_embeddings(user_id)
    if embeddings is None or not chunks:
        return []
    query_emb = get_openai_embedding(query, api_key)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(np.array([query_emb]), top_k)
    return [chunks[i] for i in I[0]]

# --- OpenAI GPT-4 Tweet Generation ---
def generate_tweet_openai(
    topic: str,
    content_snippet: str,
    custom_instructions: str,
    max_tokens: int = 60
):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = (
        f"Your task is to generate an educational, concise tweet about '{topic}'. "
        f"Use this source content: {content_snippet.strip()}\n"
        f"Additional instructions: {custom_instructions.strip()}\n"
        "The tweet should be informative, easy to read, and fit within Twitter's character limit (max 280 chars)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who writes social media posts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
        )
        tweet = response.choices[0].message.content.strip()
        return tweet
    except Exception as e:
        return f"Error generating tweet: {e}"

# --- Tweet Posting (Tweepy) ---
def post_tweet_to_x(tweet_text):
    api_key = os.getenv("X_API_KEY")
    api_secret = os.getenv("X_API_SECRET_KEY")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    try:
        api.update_status(status=tweet_text)
        return True, "Tweet posted successfully."
    except Exception as e:
        return False, str(e)
