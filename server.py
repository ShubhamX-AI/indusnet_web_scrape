import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

app = FastAPI()

# --- CONFIGURATION ---
LINKS_FILE = "intglobal_links.txt"
TOP_K_RESULTS = 3  # How many top links to scrape per query

# --- 1. PRE-PROCESSING HELPERS ---

def load_links():
    """Loads links from the text file."""
    try:
        with open(LINKS_FILE, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
        return list(set(links)) # Remove duplicates just in case
    except FileNotFoundError:
        return []

def url_to_text(url):
    """
    Converts a URL into a searchable string.
    Ex: 'https://intglobal.com/services/air-freight' -> 'services air freight'
    """
    # Remove http/https and domain
    # You might need to adjust the regex if the domain changes
    cleaned = re.sub(r'https?://(www\.)?intglobal\.com/', '', url)
    
    # Replace non-alphanumeric chars (like / - _ ) with spaces
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', cleaned)
    
    return cleaned.strip()

# --- 2. THE SEARCH ENGINE LOGIC ---

class SearchEngine:
    def __init__(self):
        self.links = load_links()
        if not self.links:
            print(f"Warning: {LINKS_FILE} is empty or missing!")
        
        # Prepare the 'corpus' (the text version of URLs)
        self.url_corpus = [url_to_text(link) for link in self.links]
        
        # Initialize Vectorizer (TF-IDF)
        # This converts text to numbers for cosine similarity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # If we have links, fit the model immediately
        if self.url_corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.url_corpus)
        else:
            self.tfidf_matrix = None

    def find_relevant_links(self, query: str, top_k=3):
        if self.tfidf_matrix is None:
            return []

        # Convert query to vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate Cosine Similarity between query and all URL texts
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices of top_k matches
        # argsort sorts ascending, so we take the end ([-top_k:]) and reverse it ([::-1])
        related_docs_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for i in related_docs_indices:
            score = cosine_similarities[i]
            # Only return if there is some similarity (score > 0)
            if score > 0.0:
                results.append((self.links[i], score))
        
        return results

# Initialize engine on startup
search_engine = SearchEngine()

# --- 3. SCRAPING LOGIC (ON DEMAND) ---

def scrape_page_content(url):
    """Visits the link and extracts the text."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Extract meaningful text (paragraphs and headers)
        # We assume the content is in <p>, <h1>-<h6>, <li>
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text = " ".join([elem.get_text(strip=True) for elem in text_elements])
        
        # Limit text length to avoid overwhelming the user (optional)
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        return f"Error scraping content: {e}"
    
    
class Citation(BaseModel):
    text: str
    url: str


class SearchResult(BaseModel):
    answer: str
    citations: List[Citation]


def scrape_using_ai(urls: List[str], query: str):
    # In a real production app, you would send 'url' to OpenAI/LLM here
    # to generate a natural language answer. For now, we return the raw data.
    
    print("Scraping using AI...")
    print("URLs:", urls)
    
    # Remove the "https://" prefix from the URLs
    urls = [url.replace("https://", "") for url in urls]
    urls = ['intglobal.com']
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    print(urls)
    
    response = client.responses.parse(
        model="gpt-5",
        reasoning={"effort": "low"},
        tools=[
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": urls,
                },
            }
        ],
        tool_choice="auto",
        include=["web_search_call.action.sources"],
        input=query,
        instructions='''"You are a corporate assistant for 'Indus Net Technologies' (intglobal.com). "
                    "You answer user queries strictly based on the provided context. "
                    "If the answer is not in the context, say so. "
                    "For every claim you make, you MUST provide a citation from the context."''',
        text_format=SearchResult,
    )

    result = response.output[-1].content[-1].parsed
    return result
    
    

# --- 4. API ENDPOINTS ---

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    print(f"Received query: {query}")

    # Step 1: Find best matching URLs based on the string similarity
    matches = search_engine.find_relevant_links(query, top_k=TOP_K_RESULTS)
    
    if not matches:
        return {"message": "No relevant links found for your query."}

    response_data = []

    # # Step 2: Scrape specific matched pages
    # for url, score in matches:
    #     print(f"Scraping matched link ({score:.2f}): {url}")
    #     content = scrape_page_content(url)
        
    #     response_data.append({
    #         "citation_url": url,
    #         "relevance_score": float(score),
    #         "scraped_content": content
    #     })
    
    # Send the urls to the ai
    response_data = scrape_using_ai([url for url, score in matches], query)

    # Step 3: Return Data
    # In a real production app, you would send 'response_data' to OpenAI/LLM here
    # to generate a natural language answer. For now, we return the raw data.
    
    return {
        "query": query,
        "results": response_data
    }

@app.get("/")
def home():
    return {"status": "Server is running. Send POST request to /ask"}