import os
import time
from pprint import pprint

import requests

api_key = os.getenv("CUSTOM_SEARCH_API_KEY")
search_engine_id = os.getenv("SEARCH_ENGINE_ID")

google_endpoint = "https://www.googleapis.com/customsearch/v1"

q = "dogs"

params = {
    'cx': search_engine_id,
    'searchType': "image",
    'key': api_key,
    'q': q,
    'num': 10,
    # 'start': #comes from subsequent calls 
}


if __name__ == "__main__":
    start_index = None

    links = [] 
   
    start = time.time()
    # for i in range(15):
    #     if i>0:
    #         params = {**params, 'start': start_index}
    #
    #     results = requests.get(google_endpoint, params=params)
    #     results = results.json()
    #     links = links + [i['link'] for i in results['items']]
    #     # next page
    #     start_index = results['queries']['nextPage'][0]['startIndex'] # pass this to next call
    #
    stop = time.time()
    print(f"outbound request time: {(stop-start)/15}/req")

    # with open("urls.txt", 'w') as f:
    #     f.writelines('\n'.join(links))

    results = requests.get(google_endpoint, params=params).json()
    pprint(results.keys())
    pprint(results)

