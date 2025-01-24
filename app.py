from fastapi import FastAPI, Request, Form, Body, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import google.generativeai as genai
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any

class ApiKeys(BaseModel):
    serpApiKey: str
    geminiApiKey: str

class GenerateRequest(BaseModel):
    input: str
    prompt: str
    section: str
    geminiApiKey: str
    serpApiKey: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

@app.post("/generate")
async def generate_content(request: GenerateRequest):
    try:
        print(f"Processing request for section: {request.section}")  # Debug log
        
        # Configure Gemini
        genai.configure(api_key=request.geminiApiKey)
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate content
        response = model.generate_content(request.prompt)
        
        if response and response.text:
            return {
                "status": "success",
                "content": response.text,
                "section": request.section
            }
        else:
            raise HTTPException(status_code=500, detail="Empty response from AI model")
            
    except Exception as e:
        print(f"Error in generate_content: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/test-api-keys")
async def test_api_keys(api_keys: ApiKeys):
    results = {
        "serpApi": False,
        "geminiApi": False
    }
    
    # Test SerpAPI
    try:
        params = {
            "engine": "google",
            "q": "test",
            "api_key": api_keys.serpApiKey,
            "num": 1
        }
        search = GoogleSearch(params)
        search.get_dict()
        results["serpApi"] = True
    except Exception as e:
        print(f"SerpAPI test failed: {e}")
    
    # Test Gemini API
    try:
        genai.configure(api_key=api_keys.geminiApiKey)
        model = genai.GenerativeModel('gemini-pro')
        model.generate_content("test")
        results["geminiApi"] = True
    except Exception as e:
        print(f"Gemini API test failed: {e}")
    
    return results

@app.post("/search")
async def search_research(
    query: str = Form(...),
    numWebsites: int = Form(10),
    contentLimit: int = Form(1000),
    promptTemplate: str = Form(...),
    serpApiKey: str = Form(...),
    geminiApiKey: str = Form(...)
):
    # Configure APIs with provided keys
    genai.configure(api_key=geminiApiKey)
    model = genai.GenerativeModel('gemini-pro')
    
    # Use SerpAPI to get search results
    params = {
        "engine": "google",
        "q": query + " research paper",
        "api_key": serpApiKey,
        "num": int(numWebsites)
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract URLs from organic results
    search_results = []
    if "organic_results" in results:
        search_results = [result["link"] for result in results["organic_results"][:int(numWebsites)]]
    
    # Scrape content from each URL
    all_content = []
    for url in search_results:
        content = scrape_website(url)
        if content:
            all_content.append(f"Content from {url}:\n{content[:int(contentLimit)]}")
    
    # Combine all content
    combined_content = "\n\n".join(all_content)
    
    # Use the custom prompt template
    formatted_prompt = promptTemplate.replace('{query}', query).replace('{content}', combined_content)
    
    # Generate research summary using Gemini
    response = model.generate_content(formatted_prompt)
    
    return {"summary": response.text, "sources": search_results} 