from serpapi import GoogleSearch

from dotenv import load_dotenv
import os
load_dotenv()

params = {
    "engine": "google_images",
    "q": "intitle: newton site:wikipedia.org",
    "api_key": os.getenv("SERPAPI_API_KEY"),
    "num": 1
}

search = GoogleSearch(params)
results = search.get_dict()
print(results["images_results"][0]["original"])