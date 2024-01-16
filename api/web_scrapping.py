import os
import serpapi
from dotenv import load_dotenv

def get_results(keyword):
    load_dotenv()

    apiKey = os.getenv('SERPAPI_KEY')
    client = serpapi.Client(api_key=apiKey)

    result = client.search(
        q=keyword,
        engine="google",
        location="Austin, Texas",
        hl="en",
        gl="us",
    )

    return result["organic_results"]
