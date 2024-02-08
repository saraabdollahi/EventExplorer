import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertConfig, BertModel, BertTokenizer
from transformers import pipeline
import torch
import itertools
import torch.nn as nn
from transformers import logging
import warnings
from torch import cuda
import random
import datetime
import json
from rank_bm25 import BM25Okapi
import sys
import gc
import pickle
import glob
import os
from sklearn.metrics import average_precision_score as map_score
from sklearn.metrics import recall_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(42)
import re
from html import unescape

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

all_queries=["great_fire_of_london"]
entities_df=pd.read_csv("/data/event_aspect_terms.tsv", sep="\t")
def remove_websites(text):
    # Regular expression pattern to match URLs and specific strings
    pattern = re.compile(r'https?://\S+|www\.\S+|\.theguardian\.com/politics/\S+')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def clean_html_text(html_text):
    unescaped_text = unescape(html_text)
    # Remove HTML tags
    cleaned_text = re.sub(r'<.*?>', '', unescaped_text)
    return cleaned_text

class diversified_ranking(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', max_length=512)
        self.BCElogit_loss=nn.BCEWithLogitsLoss()
        self.ranking_loss=nn.MarginRankingLoss(margin=0.1)
        self.max_epoch=5
        self.LR=0.001
        self.configuration=BertConfig()
        self.bert_model=BertModel(self.configuration)
        self.configuration=self.bert_model.config
        self.bertcat_linear = nn.Linear(768, 1, bias=False)
        self.model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        if self.bertcat_linear.bias is not None:
            nn.init.zeros_(self.bertcat_linear.bias)

    def final_stage(self,content_df,query,bert_model, linear, model, entity):
        query=query.replace("_"," ")
        if entity=="when":
            new_query="when did "+query+" happen?"
        elif entity=="cause":
            new_query="what was the cause of "+query+"?"
        elif entity=="result":
            new_query="what was the result of "+query+"?"
        elif entity=="event":
            new_query=query
        else:
            new_query=query+" "+entity
        tmp_bert_scores_df=self.catbert(query,new_query, content_df, model, t.bertcat_linear, entity)
        return(tmp_bert_scores_df)


    def catbert(self,query,new_query,content_df, model, bertcat_linear, entity):
        tmp_bert_scores_df=pd.DataFrame() 
        content_df["Title"]=content_df["Title"].fillna("")
        content_df["content"]=content_df.apply(lambda row: row.Title+" "+row.Snippet, axis=1)
        content_df["content"]=content_df.apply(lambda row: row.Title+" "+row.cleaned_snippet, axis=1)
                
        query_output=self.tokenizer(new_query, return_tensors="pt")
        query_tokens=self.tokenizer(new_query, return_tensors="pt")["input_ids"][0]
        query_attention=query_output["attention_mask"]
        query_attention=query_attention.to(device)
        query_tokens=query_tokens.to(device)
        
        query_segments_ids = [0] * (query_tokens.size()[0]) 
        document_segments_ids=[1] * (200)
        segments_ids=query_segments_ids+document_segments_ids
        segments_tensors = torch.tensor([segments_ids])
        segments_tensors=segments_tensors.to(device)
        catbert_results=pd.DataFrame(columns=["number","score","title", "date","original_url","archive_url", "snippet","cleaned_snippet", "embedding"])
        catbert_t1=datetime.datetime.now()
        document_embeddings=[]
        for i in range(content_df.shape[0]):
        
            with torch.no_grad():
                tokenizer_output=self.tokenizer(content_df.iloc[i]["content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
                tokenizer_output=tokenizer_output.to(device)
                content=tokenizer_output["input_ids"]
                content_attention=tokenizer_output["attention_mask"]
                doc_embedding=self.model(content)[0].mean(dim=1).detach().cpu() 
                document_embeddings.append(doc_embedding.numpy().reshape(-1))
                tmp_output=self.model(torch.cat([query_tokens.unsqueeze(0),content], dim=1), attention_mask=torch.cat([query_attention, content_attention], dim=1), token_type_ids=segments_tensors[0,:].unsqueeze(0)) 

                pos_last_hidden_states = tmp_output.last_hidden_state
                pos_score=(torch.transpose(pos_last_hidden_states[0,0,:], 0, -1))
            score=self.bertcat_linear(pos_score)
            catbert_results=catbert_results.append({"number":i, "score":score.item(),"title":content_df.iloc[i]["Title"], "date_type":content_df.iloc[i]["date_type"], "date":content_df.iloc[i]["date"],"original_url":content_df.iloc[i]["Original URL"],"archive_url":content_df.iloc[i]["Link to archive"] ,"content":content_df.iloc[i]["Snippet"],"cleaned_snippet":content_df.iloc[i]["cleaned_snippet"], "embedding":doc_embedding}, ignore_index=True)
        doc_array=np.vstack(document_embeddings)
        dissimilarity_scores=1-cosine_similarity(doc_array, doc_array)
        catbert_t2=datetime.datetime.now()
        
        '''for content diversification'''
        def calculate_similarity(row, embeddings):
            current_embedding=catbert_results.iloc[row]["embedding"]
            if row==0:
                content_diff=[[1]]
            else:
                tmp_df=catbert_results.loc[:row-1,]
                previous_embeddings=list(tmp_df["embedding"])
                content_diff=1-cosine_similarity(current_embedding, np.vstack(previous_embeddings))
            return(content_diff[0])
       
        ''' for time diversification'''
        def calculate_date_diversity(row, catbert_results):
            current_date=catbert_results.iloc[row]["date_type"]
            if row==0:
                date_dif=120 ### the highest difference for the first ranked score
            else:
                tmp_df=catbert_results.loc[:row-1,]
                previous_dates=list(tmp_df["date_type"])
                date_dif=[abs((current_date - previous_date).days) for previous_date in previous_dates]
            return(date_dif)

        scaler=MinMaxScaler()
        catbert_results=catbert_results.sort_values(by="score", ascending=False)
        catbert_results=catbert_results.reset_index(drop=True)
        catbert_results["content_diff_scores"]= catbert_results.apply(lambda row: calculate_similarity(row.name, catbert_results), axis=1)
        catbert_results["date_diff_scores"]= catbert_results.apply(lambda row: calculate_date_diversity(row.name, catbert_results), axis=1)
        catbert_results["max_similarity"]= catbert_results.apply(lambda row: np.max(row.content_diff_scores), axis=1)
        catbert_results["mean_similarity"]=catbert_results.apply(lambda row: np.mean(row.content_diff_scores), axis=1)
        catbert_results["max_date_diff"]= catbert_results.apply(lambda row: np.max(row.date_diff_scores), axis=1)
        catbert_results["mean_date_diff"]=catbert_results.apply(lambda row: np.mean(row.date_diff_scores), axis=1)
        catbert_results["ranking_score"]=scaler.fit_transform(catbert_results[["score"]])
        catbert_results["max_date_diff"]=scaler.fit_transform(catbert_results[["max_date_diff"]])
        catbert_results["mean_date_diff"]=scaler.fit_transform(catbert_results[["mean_date_diff"]])
        catbert_results["final_score1"]=0.7* catbert_results["ranking_score"]+0.15*catbert_results["mean_similarity"]+0.15*catbert_results["mean_date_diff"]
        catbert_results["final_score2"]=0.7* catbert_results["ranking_score"]+0.15*catbert_results["max_similarity"]+0.15*catbert_results["max_date_diff"]
        query_=query.replace(" ","_")
        catbert_results.to_csv("/data/events/"+query_+"/"+entity+"_diversified_scores.csv", sep=",", index=False)
        return(tmp_bert_scores_df)



t=diversified_ranking()
t.load_state_dict(torch.load("warag_monobert"))
device="cuda" if cuda.is_available() else "cpu"
t.eval()
t=t.to(device)




def ranking(all_queries, bert_model, bertcat_linear, model, bert_scores_df):
    random.shuffle(all_queries)
    device="cuda" if cuda.is_available() else "cpu" 
    for q in range(len(all_queries)):
        entities_tmp=entities_df.loc[entities_df["event_label"]==all_queries[q],]
        entities=list(entities_tmp["term"].unique())
        entities=entities+["when","cause","result","event"]
        for entity in entities:
            try:
                content_df=pd.read_csv("/data/events/"+all_queries[q]+"/"+entity+"_results.tsv", sep="\t")
                content_df["cleaned_snippet"]=content_df.apply(lambda row: clean_html_text(row.Snippet), axis=1)
                content_df["cleaned_snippet"]=content_df.apply(lambda row: remove_websites(row.Snippet), axis=1)
                content_df["date"]=content_df.apply(lambda row: str(row.Day)+"/"+str(row.Month)+"/"+str(row.Year), axis=1)
                content_df["year"]=content_df["Year"].astype(int)
                
                content_df["date_type"]=pd.to_datetime(content_df["date"])
                content_df["Title"]=content_df["Title"].fillna("")
                
                content_df["tmp"]=content_df.apply(lambda row: row.Title+" "+row.Snippet, axis=1)
                content_df=content_df.sort_values(by="date_type").drop_duplicates("tmp", keep="first")
                del content_df["tmp"]
                tmp_bert_scores_df=t.final_stage(content_df, all_queries[q], bert_model, bertcat_linear, model, entity)
                log_file.flush()
                bert_scores_df=bert_scores_df.append(tmp_bert_scores_df)
            except:
                print("exception for: ", all_queries[q], entity)
    return (bert_scores_df)
    
def main(all_queries):
    bert_scores_df=pd.DataFrame()
    new_bert_results=ranking(all_queries, t.bert_model, t.bertcat_linear, t.model, bert_scores_df)


if __name__=="__main__":
    main(all_queries)
