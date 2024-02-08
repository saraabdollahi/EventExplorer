#  WA-RAG, an Interactive System for Exploring Event Collections

Welcome to the GitHub repository for the paper titled 'Can we bring together Manual Curation of Web Archive Collections and Automatic Creation? Requirements, Methods and Challenges. This repository contains the implementation code for the WA-RAG model. Please note that author names and specific details cannot be disclosed due to the anonymous submission.


## Repository Structure

* #### web_archive_content_preprocessing.py

      The scripts for preprocessing exported results from the Portuguese Web Archive (PWA). These scripts detect languages and keep only English data.

* #### warag_monobert_training.py

      Fine-tuning BERT on the MS-MARCO event dataset. The training involves reformulated questions using annotated aspects and question templates.

* #### diversified_ranking.py

      The diversified_ranking script ranks snippets from web archives. It uses a trained ranking model and considers event aspects, text diversity, and temporal diversity. 

* #### data folder

      The data folder contains example datasets that can be used to run the scripts in this repository.
