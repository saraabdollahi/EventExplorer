#  EventExplorer, an Interactive System for Exploring Event Collections

Welcome to the GitHub repository for the paper titled "Retrieval-Augmented Generation of Event Collections from
Web Archives and the Live Web". This repository contains the implementation code for the EventExplorer model. 

## Repository Structure

* #### event_aspect_retrieval.py

      Given a list of events, this script extracts its aspects such as relevant participants,  and locations and ranks them according to the number of links to the event on EventKG.

* #### web_archive_content_preprocessing.py

      The scripts for preprocessing exported results from the Portuguese Web Archive (PWA). These scripts detect languages and keep only English data.

* #### warag_monobert_training.py

      Fine-tuning BERT on the MS-MARCO-Event dataset. The training involves reformulated questions using annotated aspects and question templates.

* #### diversified_ranking.py

      The diversified_ranking script ranks snippets from web archives. It uses a trained ranking model and considers event aspects, text diversity, and temporal diversity. 

* #### Data folder

      The data folder contains example datasets that can be used to run the scripts in this repository.
