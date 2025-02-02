#  EventExplorer, an Interactive System for Exploring Event Collections

Welcome to the GitHub repository for the paper titled "Retrieval-Augmented Generation of Event Collections from
Web Archives and the Live Web". This repository contains the implementation code for the EventExplorer model. 



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

1. #### event_aspect_retrieval.py

  Given a list of events, this script extracts its aspects such as relevant participants,  and locations and ranks them according to the number of links to the event on EventKG.

📥 Input:
The script starts with a predefined list of events:
 ```sh
queries = ["event_662534"]
```
You can modify this list in the script to include your own events.

📤 Output:
A TSV file containing ranked aspects:
```sh
./data/event_links_cnts.tsv
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
  ./data/{event}/{entity}_results.tsv
  ```

* #### eventexplorer_monobert_training.py

  Fine-tuning BERT on the MS-MARCO-Event dataset. The training involves reformulated questions using annotated aspects and question templates.
  The script saves the trained model after the completing the training process named warag_monobert at the current directory. 

* #### diversified_ranking.py

      The diversified_ranking script ranks snippets from web archives. It uses a trained ranking model and considers event aspects, text diversity, and temporal diversity.
  ##### Inputs
- **`./data/event_aspect_terms.tsv`**: This file contains the event-aspect terms used to generate queries related to an event's aspects.
- **`warag_monobert`**: A pre-trained BERT model used for document ranking. It is fine-tuned to handle event-related queries.

##### Outputs
- **Ranked Document Scores**: The final rankings for each event and entity are output as `.csv` files, stored in the `./data/` directory. Each file contains the ranking scores of documents related to the event-aspect pairs, along with additional features such as content and date diversity scores.


* #### Data folder

      The data folder contains example datasets that can be used to run the scripts in this repository.
