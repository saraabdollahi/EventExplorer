import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import warnings
import datetime
warnings.filterwarnings("ignore")


all_events=["Grenfell Tower fire"]
entities_df=pd.read_csv("./data/event_aspect_terms.tsv", sep="\t")
event_ids=pd.read_csv("./data/event_ids.tsv", sep="\t")
event_ids=event_ids.rename(columns={"eventkg_id":"event", "label":"event_label"})
entities_df=pd.merge(left=entities_df, right=event_ids, how="left", on="event")
entities_df["event_label"]=entities_df["event_label"].str.lower()
entities_df["event_label"]=entities_df["event_label"].str.replace(" ","_")


def get_publication_date_from_url(url):
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.split('/')
    date_regex = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{4}/\d{2}/\d{2})")
    for segment in path_segments:
        match = date_regex.search(segment)
        if match:
            return match.group()

def search_bing_multiple_pages(timeframe, query, api_key, num_results_per_page=10, num_pages=10):
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }
    search_results = []

    for page_num in range(num_pages):
        params = {
            "q": query,
            "count": num_results_per_page,
            "offset": page_num * num_results_per_page,
            "mkt": "en-EU",
            "responseFilter": "Webpages",
            "safeSearch": "Strict",
            "freshness":timeframe

        }

        try:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if "webPages" in data and "value" in data["webPages"]:
                for item in data["webPages"]["value"]:
                    #print (item)
                    title = item.get("name", "")
                    snippet = item.get("snippet", "")
                    url = item.get("url", "")
                    publication_date = get_publication_date_from_url(url)
                    date=item.get("datePublished","")
                    search_results.append({"title": title, "snippet": snippet, "url": url, "publication_date": date})

        except requests.RequestException as e:
            print(f"Error fetching search results for page {page_num}: {e}")

    return search_results
api_key = "your-api-key-here"
num_results_per_page = 100
num_pages = 10





for event in all_events:
    event_begintime=datetime.datetime.now()
    tmp_entities=entities_df.loc[entities_df["event_label"]==event.lower(),]
    tmp_entities=tmp_entities.reset_index(drop=True)
    timeframe=tmp_entities.iloc[0]["timeframe"]
    timeframe="2017-06-14..2017-10-14" ### setting the timeframe 4 month after the happening date
    entities=list(tmp_entities["term"].unique())
    entities=entities+["when","cause","result","event"]

    for entity in entities:    		
        if entity=="result":
            query = "What was the result of "+event.replace("_"," ")+"?"
        elif entity=="cause":
            query = "What was the cause of "+event.replace("_"," ")+"?"
        elif entity=="when":
            query = "When did "+event.replace("_"," ")+" happen?"
        elif entity=="event":
            query = event.replace("_"," ")
        else:
            query = event.replace("_"," ")+" "+entity
       
        df=pd.DataFrame()
        results = search_bing_multiple_pages(timeframe, query, api_key, num_results_per_page, num_pages)

        for index, result in enumerate(results, start=1):
            df=df.append({"title":result["title"], "snippet":result["snippet"], "url":result["url"], "date":result["publication_date"]}, ignore_index=True)
            label="./data/"+event.lower()+"/"+entity+"_bing_results.csv"
            df.to_csv(label, sep=",")
    event_endtime=datetime.datetime.now()
    print("The whole event retrieval took: ", (event_endtime-event_begintime).total_seconds())
