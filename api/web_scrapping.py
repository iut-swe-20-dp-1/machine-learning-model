# import os
# import serpapi
# from dotenv import load_dotenv

# def get_results(keyword):
#     load_dotenv()

#     apiKey = os.getenv('SERPAPI_KEY')
#     client = serpapi.Client(api_key=apiKey)

#     result = client.search(
#         q=keyword,
#         engine="google",
#         location="Austin, Texas",
#         hl="en",
#         gl="us",
#     )

#     return result["organic_results"]



params = {
  "api_key": "9409b23e7455a9cc1a233abbbba6ab7f2176e927cad4f1c34f38bda644bbf6eb",
  "engine": "google",
  "q": "ways to relieve academic stress",
  "location": "Austin, Texas, United States",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en",
  "safe": "active"
}

search = GoogleSearch(params)
results = search.get_dict()
print(results)
