#  EventExplorer, an Interactive System for Exploring Event Collections

Welcome to the GitHub repository for the paper titled "Retrieval-Augmented Generation of Event Collections from
Web Archives and the Live Web". This repository contains the implementation code for the EventExplorer model. 

The code to deploy the EventExplorer website is available in [another GitHub repository](https://github.com/sgottsch/EventExplorerWebsite).

## 🚀 Installation Guide

First, clone the repository and navigate into the project folder:

```sh
git clone https://github.com/saraabdollahi/EventExplorer.git
cd EventExplorer
```
Then make sure you have Python installed. Then install the required packages:
```sh
pip install -r requirements.txt
```

## 📁 Repository Structure

* #### event_aspect_retrieval.py

  Given a list of events, this script extracts its aspects such as relevant participants,  and locations and ranks them according to the number of links to the event on EventKG.

  📥 Input:
  The script starts with a predefined list of events:
   ```sh
  queries = ["event_662534"]
  ```
  You can modify this list in the script to include your own events.

  📤 Output:
  A .tsv file containing ranked aspects:
  ```sh
  ./data/event_links_cnts.tsv
  ```

* ####  bing_api_search.py  

  This script retrieves search results from Bing for a given event and related entities (aspects), extracting relevant web pages and metadata.  

  📤 Output: 
    Search results stored as .csv files at:
    ```sh
    ./data/{event}/{entity}_bing_results.csv
    ```


* #### web_archive_content_preprocessing.py

  The scripts for preprocessing exported results from the Portuguese Web Archive (PWA). These scripts detect languages and keep only English data.
  
  📥 Input: 
  The raw snippets from web archives for a give event and its aspect (entity), located at:
  ```sh
  ./data/{event}/{entity}.csv
  ```
  📤 Output: 
  Processed snippets stored as .tsv file at:
  ```sh
  ./data/{event}/{entity}_PWA_results.tsv
  ```

* #### eventexplorer_monobert_training.py

  Fine-tuning BERT on the MS-MARCO-Event dataset. The training involves reformulated questions using annotated aspects and question templates.
  
  📤 Output: 
  Trained model (event_explorer_monobert) saved in the current directory after training.

* #### diversified_ranking.py

      The diversified_ranking script ranks snippets from web archives. It uses a trained ranking model and considers event aspects, text diversity, and temporal diversity.
  
  📥 Input:
  Event-aspect terms used to generate queries related to an event's aspects:
  ```sh
   event_explorer_monobert: The trained model from eventexplorer_monobert_training.py
  ```

  📤 Output:
  The final rankings for each event and entity (the corresponding aspect):
  ```sh
  ./data/{event}/{entity}_diversified_scores.csv.tsv
  ```

* #### component_generation.py
  This script generates **descriptive components** for event collections using a **Retrieval-Augmented Generation (RAG) approach**. It utilizes **ChatGPT** to generate summary, metadata and timeline of an event based on top-ranked documents. 

  📥 Input:
    Ranking of documents per event and related entity (aspect) located at:
    ```sh
     ./data/{event}/{entity}_diversified_scores.csv
    ```

  📤 Output:
    Generated Descriptive Components (saved as JSON) for all the aspects of an event stored at: 
    ```sh
    ./data/{event}/rag_results.json 
    ```

* #### Data folder

      The data folder contains example datasets that can be used to run the scripts in this repository.



## 🔑 API Keys Required  

Two scripts in this repository require API keys for execution:  

1. **component_generation.py** – Requires an **OpenAI API key** for generating descriptive components using ChatGPT.  
   - Define OpenAI API key  directly in the script where needed  
     

2. **bing_api_search.py** – Requires a **Bing Search API key** to retrieve search results from Bing.  
   - Specify this key inside the script.  

Make sure you **replace `"your-api-key-here"` with your actual API keys** before running these scripts.  

# 📧 Contact

Sara Abdollahi ([abdollahi@L3S.de](mailto:abdollahi@L3S.de)) & Simon Gottschalk ([gottschalk@L3S.de](mailto:gottschalk@L3S.de))

# Reference

To be announced.
