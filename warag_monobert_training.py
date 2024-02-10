import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertConfig, BertModel, BertTokenizer
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
import chardet


torch.manual_seed(42)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
all_df=pd.read_csv("./data/query_aspect_docs.tsv", sep="\t")
data=pd.read_csv("./data/collection.tsv", sep="\t", header=None, names=["col1","col2"])
data=data.rename(columns={"col1":"doc_id", "col2":"content"})
all_queries=list(all_df["event"].unique())
event_terms=pd.read_csv("./data/event_aspect_terms.tsv", sep="\t")
event_terms["cnt"]=event_terms["cnt"].astype(int)
event_terms["rank"]=event_terms.groupby(["event"])["cnt"].rank("first", ascending=False)
event_terms=event_terms.rename(columns={"event":"eventkg_id"})
event_ids=pd.read_csv("./data/event_ids.tsv", sep="\t")
event_ids=event_ids.rename(columns={"label":"event"})
event_ids["event"]=event_ids["event"].str.lower()
event_terms=pd.merge(left=event_terms, right=event_ids, how="left", on="eventkg_id")
event_terms=event_terms.loc[event_terms["rank"]<11,]




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


    def final_stage(self,content_df, query, bert_model, linear, model, nugget, term):
        all_pos_neg_results, all_pos_neg_labels=self.catbert(query, content_df, model, t.bertcat_linear, nugget, term)
        loss=self.BCElogit_loss(all_pos_neg_results, all_pos_neg_labels) 
        return(loss)

    def catbert(self,query, content_df, model, bertcat_linear, nugget, term):
        print("in catbert")
        if term=="":
            new_query=query+" "+nugget
        else:
            new_query=query+" "+term
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
        pos_df=content_df[["event","pos_doc","pos_content"]].drop_duplicates().reset_index(drop=True)
        pos_catbert_results=pd.DataFrame(columns=["doc_id","score","label"]) 
        pos_tensor=torch.tensor([0.0], requires_grad=True)
        pos_tensor=pos_tensor.to(device)
        for i in range(pos_df.shape[0]):
            with torch.no_grad():
               tokenizer_output=self.tokenizer(pos_df.iloc[i]["pos_content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
               tokenizer_output=tokenizer_output.to(device)
               content=tokenizer_output["input_ids"]
               content_attention=tokenizer_output["attention_mask"]
               tmp_output=self.model(torch.cat([query_tokens.unsqueeze(0),content], dim=1), attention_mask=torch.cat([query_attention, content_attention], dim=1), token_type_ids=segments_tensors[0,:].unsqueeze(0)) 

               pos_last_hidden_states = tmp_output.last_hidden_state
               pos_score=(torch.transpose(pos_last_hidden_states[0,0,:], 0, -1))
            score=self.bertcat_linear(pos_score)
            pos_tensor=torch.cat((pos_tensor, score), dim=0)
            pos_catbert_results=pos_catbert_results.append({"doc_id":pos_df.iloc[i]["pos_doc"], "score":score.item(), "label":1}, ignore_index=True)
        neg_df=content_df[["event","neg_doc","neg_content"]].drop_duplicates().reset_index(drop=True)
        neg_tensor=torch.tensor([0.0], requires_grad=True)
        neg_tensor=neg_tensor.to(device)
        for i in range(neg_df.shape[0]):
            with torch.no_grad():
               tokenizer_output=self.tokenizer(neg_df.iloc[i]["neg_content"], return_tensors="pt", max_length=200, padding="max_length", truncation="only_first")
               tokenizer_output=tokenizer_output.to(device)
               content=tokenizer_output["input_ids"]
               content_attention=tokenizer_output["attention_mask"]
               tmp_output=self.model(torch.cat([query_tokens.unsqueeze(0),content], dim=1), attention_mask=torch.cat([query_attention, content_attention], dim=1), token_type_ids=segments_tensors[0,:].unsqueeze(0)) 
               neg_last_hidden_states = tmp_output.last_hidden_state
               neg_score=(torch.transpose(neg_last_hidden_states[0,0,:], 0, -1))
            score=self.bertcat_linear(neg_score)
            neg_tensor=torch.cat((neg_tensor, score), dim=0)
            pos_catbert_results=pos_catbert_results.append({"doc_id":neg_df.iloc[i]["neg_doc"], "score":score.item(), "label":0}, ignore_index=True)
        neg_df=pos_catbert_results.loc[pos_catbert_results["label"]==1,]
        neg_min=np.min(neg_df["score"])
        pos_df=pos_catbert_results.loc[pos_catbert_results["label"]==0,]
        neg_min=np.min(neg_df["score"])
        pos_min=np.min(pos_df["score"])
        neg_scores=neg_tensor[1:]
        pos_scores=pos_tensor[1:]
        if neg_min>pos_min:
            min_=pos_min
        else:
            min_=neg_min 
        neg_max=np.max(neg_df["score"])
        pos_max=np.max(pos_df["score"])
        if neg_max>pos_max:
            max_=neg_max
        else:
            max_=pos_max
        neg_scores_=(neg_scores-min_)/(max_-min_)
        pos_scores_=(pos_scores-min_)/(max_-min_)
        pos_labels=torch.ones(pos_tensor[1:].size()[0], requires_grad=True)
        neg_labels=torch.zeros(neg_tensor[1:].size()[0], requires_grad=True)
        all_pos_neg_labels=torch.cat((pos_labels, neg_labels), -1)
        all_pos_neg_results=torch.cat((pos_scores, neg_scores), dim=0) 
        all_pos_neg_labels=all_pos_neg_labels.to(device)
        all_pos_neg_results=all_pos_neg_results.to(device)
        return(all_pos_neg_results, all_pos_neg_labels)

t=diversified_ranking()
device="cuda" if cuda.is_available() else "cpu"
t=t.to(device)

for param in t.parameters():
    param.requires_grad = True


def train_iteration(all_queries, bert_model, bertcat_linear, model, optimizer):
    t.train()
    random.shuffle(all_queries)
    total_loss=torch.tensor(0.0)
    device="cuda" if cuda.is_available() else "cpu" 
    total_loss=total_loss.to(device)
    total_final_stage_loss=0
    
    all_df=pd.read_csv("./data/query_aspect_docs.tsv", sep="\t")
    all_df=all_df.loc[all_df["aspect"].isin(["where","who","when","cause","result","other"]),]
    all_df=all_df.loc[all_df["event"].isin(all_queries),]
    all_df=all_df.loc[all_df["aspect"].notna(),]
    all_df=all_df.rename(columns={"pos_docs":"pos_doc", "neg_docs":"neg_doc"})
    for q in range(len(all_queries)):
        try:
                optimizer.zero_grad()
                event_terms_tmp=event_terms.loc[event_terms["event"]==all_queries[q],]
                loss=torch.tensor(0.0)
                loss=loss.to(device)
                df_tmp=all_df.loc[all_df["event"]==all_queries[q],]
                nugget_list=list(df_tmp["aspect"].unique())
                event_terms_tmp=event_terms.loc[event_terms["event"]==all_queries[q],]
                for nugget in nugget_list:
                    nugget_terms=event_terms_tmp.loc[event_terms_tmp["aspect"]==nugget,]
                    if nugget_terms.empty and nugget!="other":
                        df_tmp1=df_tmp.loc[df_tmp["aspect"]==nugget,]
                        df_tmp1=df_tmp1.explode("pos_doc")
                        df_tmp1=df_tmp1.explode("neg_doc")
                        nugget_content=pd.merge(left=df_tmp1, right=data, how="left", left_on="pos_doc", right_on="doc_id")
                        nugget_content=nugget_content.rename(columns={"content":"pos_content"})
                        nugget_content=pd.merge(left=nugget_content, right=data, how="left", left_on="neg_doc", right_on="doc_id")
                        nugget_content=nugget_content.rename(columns={"content":"neg_content"})
                        final_stage_loss=t.final_stage(nugget_content,all_queries[q], bert_model, bertcat_linear, model, nugget, "")
                        total_final_stage_loss+=final_stage_loss
                        final_stage_loss.backward()
                        optimizer.step()
                        final_stage_loss_score=final_stage_loss.detach()
                        final_stage_loss_score=final_stage_loss_score.to("cpu")
                        
                    else:
                        terms_list=list(nugget_terms["term"].unique())
                        if len(terms_list)>0:
                            for term in terms_list:
                                df_tmp2=df_tmp.loc[df_tmp["aspect"]==nugget,]
                                df_tmp2=df_tmp2.explode("pos_doc")
                                df_tmp2=df_tmp2.explode("neg_doc")
                                nugget_content=pd.merge(left=df_tmp2, right=data, how="left", left_on="pos_doc", right_on="doc_id")
                                nugget_content=nugget_content.rename(columns={"content":"pos_content"})
                                nugget_content=pd.merge(left=nugget_content, right=data, how="left", left_on="neg_doc", right_on="doc_id")
                                nugget_content=nugget_content.rename(columns={"content":"neg_content"})
                                nugget_content=nugget_content.loc[nugget_content["neg_content"].notna(),]
                                final_stage_loss=t.final_stage(nugget_content,all_queries[q], bert_model, bertcat_linear, model, nugget, term)
                                total_final_stage_loss+=final_stage_loss
                                final_stage_loss.backward()
                                optimizer.step()
                                final_stage_loss_score=final_stage_loss.detach()
                                final_stage_loss_score=final_stage_loss_score.to("cpu")
        except:
            print("exception for query: ", all_queries[q])
    return (total_final_stage_loss, total_loss)

def main(all_queries):
    optimizer=torch.optim.Adam(t.parameters(),lr=t.LR)
    print("Epoch zero starts")
    total_final_stage_loss, loss=train_iteration(all_queries, t.bert_model, t.bertcat_linear, t.model, optimizer)
    for epoch in range(1,t.max_epoch):
        print("Epoch ",epoch, " starts")
        total_final_stage_loss, loss=train_iteration(all_queries, t.bert_model, t.bertcat_linear, t.model,optimizer)  
    torch.save(t.state_dict(),"warag_monobert")


if __name__=="__main__":
    main(all_queries)
