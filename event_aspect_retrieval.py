import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import requests
import time
pd.options.mode.chained_assignment = None
import json
from SPARQLWrapper import SPARQLWrapper, JSON

'''This function gets the number of relations for an event and en entity pair'''
def get_count(e,t):
    label='"'+e+'"'
    rq="""
PREFIX eventKG-s: <http://eventKG.l3s.uni-hannover.de/schema/>
SELECT (SUM(?a) AS ?cnt)  WHERE {{
			{{#?object rdfs:label "{1}"@en.
              ?relation2 rdf:subject eventkg-r:{0}  .
                ?relation2 rdf:object eventkg-r:{1} .
         ?relation2 eventkg-s:links ?a.             
            }}
  UNION {{ ?object rdfs:label "{1}"@en.
    ?relation1 rdf:subject eventkg-r:{1}  .
				?relation1 rdf:object eventkg-r:{0} .
         ?relation1 eventkg-s:links ?a.
        }}
}}
GROUP BY ?object 
	"""
    sparql = "https://eventkginterface.l3s.uni-hannover.de/sparql"  
    r = requests.get(sparql, params = {'format': 'json', 'query': rq.format(e, t)})
    data = r.json()
    if (pd.io.json.json_normalize(data['results']['bindings']).empty):
        return False
    else:
        result=pd.io.json.json_normalize(data['results']['bindings'])
        result=result.rename(columns={"cnt.value":"cnt"})
        result["event"]=e
        result["term"]=t
        result=result[["event","term","cnt"]]            
    return result.iloc[0]["cnt"]



'''This function extracts actors from text subevents '''
def get_text_actor(e):
    label='"'+e+'"'
    rq="""
SELECT DISTINCT ?actor ?actor_label
WHERE
{{
eventkg-r:{0} sem:hasSubEvent ?event_sub. 
?event_sub rdf:type eventkg-s:TextEvent.
?event_sub sem:hasActor ?actor.
?actor rdfs:label ?actor_label.
filter (lang (?actor_label)="en").
}}
	"""
    sparql = "https://eventkginterface.l3s.uni-hannover.de/sparql"  
    r = requests.get(sparql, params = {'format': 'json', 'query': rq.format(e)})
    data = r.json()
    if (pd.io.json.json_normalize(data['results']['bindings']).empty):
        return False
    else:
        result=pd.io.json.json_normalize(data['results']['bindings'])
        result=result.rename(columns={"actor.value":"actor", "actor_label.value":"actor_label"})
        result["event"]=e
        result=result[["event","actor", "actor_label"]]               
    return result


'''This function extracts subevents for an event'''
def get_subevents(e):
    label='"'+e+'"'
    rq="""
SELECT DISTINCT ?event_sub ?sub_label 
WHERE
{{
eventkg-r:{0} sem:hasSubEvent ?event_sub.  
  ?event_sub rdf:type sem:Event.
  ?event_sub rdfs:label ?sub_label.
  filter (lang (?sub_label)="en").
}}
	"""
    sparql = "https://eventkginterface.l3s.uni-hannover.de/sparql"  
    r = requests.get(sparql, params = {'format': 'json', 'query': rq.format(e)})
    data = r.json()
    if (pd.io.json.json_normalize(data['results']['bindings']).empty):
        return False
    else:
        result=pd.io.json.json_normalize(data['results']['bindings'])
        result=result.rename(columns={"event_sub.value":"sub_event_id", "sub_label.value":"sub_event"})
        result["event"]=e
        result=result[["event","sub_event_id","sub_event"]]
    return result

'''This function extracts places for an event'''
def get_place(e):
    label='"'+e+'"'
    rq="""
SELECT DISTINCT ?place ?place_label
WHERE
{{
eventkg-r:{0} sem:hasPlace ?place.
?place rdfs:label ?place_label
filter (lang(?place_label)="en")
 }}
	"""
    sparql = "https://eventkginterface.l3s.uni-hannover.de/sparql"  
    r = requests.get(sparql, params = {'format': 'json', 'query': rq.format(e)})
    data = r.json()
    if (pd.io.json.json_normalize(data['results']['bindings']).empty):
        return False
    else:
        result=pd.io.json.json_normalize(data['results']['bindings'])
        result=result.rename(columns={"place.value":"place_id", "place_label.value":"place_label"})
        result["event"]=e
        result=result[["event","place_id", "place_label"]]
    return result
	
'''This function extracts related entities from Relations on EventKG for an event'''
def get_relation_terms(e):
    label='"'+e+'"'
    rq="""
	SELECT DISTINCT ?tail ?label
WHERE 
{{
?r rdf:subject eventkg-r:{0}.
?r rdf:object ?tail.
?tail rdfs:label ?label.
Filter (CONTAINS(lcase(str(?r)), "resource/relation")).
Filter(lang(?label)="en").
}}
	"""
    sparql = "https://eventkginterface.l3s.uni-hannover.de/sparql"  
    r = requests.get(sparql, params = {'format': 'json', 'query': rq.format(e)})
    data = r.json()
    if (pd.io.json.json_normalize(data['results']['bindings']).empty):
        return False
    else:
        result=pd.io.json.json_normalize(data['results']['bindings'])
        result=result.rename(columns={"tail.value":"tail","label.value":"label"})
        result["event"]=e
        result=result[["event","tail","label"]]
    return result

'''The list of input events to get expansion terms'''
queries=["event_662534"]

results1=pd.DataFrame(columns=["event","entity","term"])
for i in range(len(queries)):
    try:
        tmp_df=get_text_actor(queries[i])
        results1=results1.append(tmp_df)
    except:
        print("exception")	

results2=pd.DataFrame(columns=["event","entity","term"])
for i in range(len(queries)):
    try:
        tmp_df=get_subevents(queries[i])
        results2=results2.append(tmp_df)
    except:
        print("exception")		

results3=pd.DataFrame(columns=["event","entity","term"])
for i in range(len(queries)):
    try:
        tmp_df=get_place(queries[i])
        results3=results3.append(tmp_df)
    except:
        print("exception")


results4=pd.DataFrame(columns=["event","entity","term"])
for i in range(len(queries)):
    try:
        tmp_df=get_relation_terms(queries[i])
        results4=results4.append(tmp_df)
    except:
        print("exception")

results1.columns=["event","entity","term"]
results2.columns=["event","entity","term"]
results3.columns=["event","entity","term"]
results4.columns=["event","entity","term"]


event_data=final_results1.append(final_results2.append(final_results3.append(final_results4)))
event_data=results1.append(results3.append(results4))


event_data["entity"]=event_data.apply(lambda row: row.entity[row.entity.rfind("/")+1:], axis=1)
event_data["cnt"]=0
event_data=event_data.loc[event_data["event"].notna(),]

event_data=event_data.reset_index(drop=True)
for i in range(event_data.shape[0]):
        t=event_data.iloc[i]["entity"]
        e=event_data.iloc[i]["event"]
    try:
        tmp_df=get_count(e,t)
        event_data.loc[i,"cnt"]=tmp_df
    except:
        print("exception")

event_data["cnt"]=event_data["cnt"].astype(str)
event_data.loc[event_data["cnt"]=="False","cnt"]="0"
event_data["cnt"]=event_data["cnt"].astype(int)
event_data=event_data.sort_values(by="cnt", ascending=False)
event_data["rank"]=1
event_data.to_csv("./data/event_links_cnts.tsv", sep="\t", index=False)
