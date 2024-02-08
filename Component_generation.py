import pandas as pd
import numpy as np
import json
import openai
from packaging import version
required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

openai.api_key = '' ### your key

summary_general_prompt_='''Follow my instructions as precisely as possible. Only provide the requested output, nothing more.

I am giving you the name of and event and a JSON list of news articles about the event.

Your task is to provide a summary of all the articles (from 1 to 10) that should focus on the event. You can use all articles. Provide references to the articles using their identifiers. Only use information that is contained in the articles.

This is an example:

=== Event and Aspect ===
Event: 2017 London Bridge attack

=== News Articles ===
{
	"1": {
		"Title": "Sputnik International - Breaking News & Analysis - Radio, Photos, Videos, Infographics",
		"Snippet": "2017 Persian Gulf Disarray: Arab States Sever Relations With Qatar London Bridge and Borough Market ...  Software Vendors 13:17 Moroccan-Italian Youssef Zaghba Named as Third Perpetrator of London Attack 13 ... -word and Kathy Griffin&#8217;s Trump severed head mock-up are connected. 06 June 2017 Will London Attacks ...",
		"Crawl Date": "6/June/2017"
	},
	"2": {
		"Title": "Twilight Language: London Bridge Attack",
		"Snippet": "). Saturday, June 03, 2017 London Bridge Attack A white van mowed down pedestrians as it sped down London ...  Twitter Share to Facebook Share to Pinterest Labels: Borough Market , Jupiter , London Bridge ... Twilight Language: London Bridge Attack Twilight Language The twilight language explores hidden ... "},
		"Crawl Date": "2/August/2017"
	"10": {
		"Title": "World Bulletin / Europe",
		"Snippet": "-attacks Sun, 04 Jun 2017 10:28:05 GMT &acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market Europe ... -after-london-bridge-attack Mon, 05 Jun 2017 12:09:41 GMT No 'direct' proof Russia meddled in US ... &quot; title=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market&quot; alt=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike ... ",
		"Crawl Date": "8/June/2017"},
	"5": {
		"Title": "Scottish and Scots Diaspora News Feed",
		"Snippet": "-A3A116A60FDD Sun, 4 Jun 2017 11:03:56 -0400 London Bridge attack treated as terrorist incident Police have confirmed that incidents at London Bridge and nearby Borough Market are terrorist incidents  3870FDAB ... ",
		"Crawl Date": "6/June/2017"
	}
}

=== Summary ===
	
The 2017 London Bridge attack and the attack on the Borough Market were treated as terrorist incidents (5). They occurred on June 3 and involved a white van that mowed down pedestrians on London Bridge (1,2). Youssef Zaghba was identified as the third perpetrator of the attack (10). The incident sparked disarray and severed relations with Qatar in the Persian Gulf region (1).

=== End of Summary ===

This is your task:

=== Event ==='''




summary_aspect_prompt_='''Follow my instructions as precisely as possible. Only provide the requested output, nothing more.

I am giving you the name of and event, an aspect of the event that you should focus on, and a JSON list of news articles about the event and its aspect.

Your task is to provide a summary of all the articles (from 1 to 10) that should focus on the event and its aspect. You can use all articles. Provide references to the articles using their identifiers. Only use information that is contained in the articles.

This is an example:

=== Event and Aspect ===
Event: 2017 London Bridge attack
Aspect: Borough Market

=== News Articles ===
{
	"1": {
		"Title": "Sputnik International - Breaking News & Analysis - Radio, Photos, Videos, Infographics",
		"Snippet": "2017 Persian Gulf Disarray: Arab States Sever Relations With Qatar London Bridge and Borough Market ...  Software Vendors 13:17 Moroccan-Italian Youssef Zaghba Named as Third Perpetrator of London Attack 13 ... -word and Kathy Griffin&#8217;s Trump severed head mock-up are connected. 06 June 2017 Will London Attacks ...",
		"Crawl Date": "6/June/2017"
	},
	"2": {
		"Title": "Twilight Language: London Bridge Attack",
		"Snippet": "). Saturday, June 03, 2017 London Bridge Attack A white van mowed down pedestrians as it sped down London ...  Twitter Share to Facebook Share to Pinterest Labels: Borough Market , Jupiter , London Bridge ... Twilight Language: London Bridge Attack Twilight Language The twilight language explores hidden ... "},
		"Crawl Date": "2/August/2017"
	"10": {
		"Title": "World Bulletin / Europe",
		"Snippet": "-attacks Sun, 04 Jun 2017 10:28:05 GMT &acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market Europe ... -after-london-bridge-attack Mon, 05 Jun 2017 12:09:41 GMT No 'direct' proof Russia meddled in US ... &quot; title=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market&quot; alt=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike ... ",
		"Crawl Date": "8/June/2017"},
	"5": {
		"Title": "Scottish and Scots Diaspora News Feed",
		"Snippet": "-A3A116A60FDD Sun, 4 Jun 2017 11:03:56 -0400 London Bridge attack treated as terrorist incident Police have confirmed that incidents at London Bridge and nearby Borough Market are terrorist incidents  3870FDAB ... ",
		"Crawl Date": "6/June/2017"
	}
}

=== Summary ===
	
The 2017 London Bridge attack and the attack on the Borough Market were treated as terrorist incidents (5). They occurred on June 3 and involved a white van that mowed down pedestrians on London Bridge (1,2). Youssef Zaghba was identified as the third perpetrator of the attack (10). The incident sparked disarray and severed relations with Qatar in the Persian Gulf region (1).

=== End of Summary ===

This is your task:

=== Event ==='''


metadata_prompt_='''Follow my instructions as precisely as possible. I am giving you a JSON list with news articles about an event. Your task is to provide a metadata JSON which consists of the date range of the event reported in the articles, the event locations mentioned in the articles and the event subjects mentioned in the articles. You can use all articles.

This is an example:

=== Event ===
Event: 2017 London Bridge attack

=== News Articles ===
{
	"1": {
		"Title": "Sputnik International - Breaking News & Analysis - Radio, Photos, Videos, Infographics",
		"Snippet": "2017 Persian Gulf Disarray: Arab States Sever Relations With Qatar London Bridge and Borough Market ...  Software Vendors 13:17 Moroccan-Italian Youssef Zaghba Named as Third Perpetrator of London Attack 13 ... -word and Kathy Griffin&#8217;s Trump severed head mock-up are connected. 06 June 2017 Will London Attacks ...",
		"Crawl Date": "6/June/2017"
	},
	"2": {
		"Title": "Twilight Language: London Bridge Attack",
		"Snippet": "). Saturday, June 03, 2017 London Bridge Attack A white van mowed down pedestrians as it sped down London ...  Twitter Share to Facebook Share to Pinterest Labels: Borough Market , Jupiter , London Bridge ... Twilight Language: London Bridge Attack Twilight Language The twilight language explores hidden ... "},
		"Crawl Date": "2/August/2017"
	"10": {
		"Title": "World Bulletin / Europe",
		"Snippet": "-attacks Sun, 04 Jun 2017 10:28:05 GMT &acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market Europe ... -after-london-bridge-attack Mon, 05 Jun 2017 12:09:41 GMT No "direct" proof Russia meddled in US ... &quot; title=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market&quot; alt=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike ... ",
		"Crawl Date": "8/June/2017"},
	"5": {
		"Title": "Scottish and Scots Diaspora News Feed",
		"Snippet": "-A3A116A60FDD Sun, 4 Jun 2017 11:03:56 -0400 London Bridge attack treated as terrorist incident Police have confirmed that incidents at London Bridge and nearby Borough Market are terrorist incidents  3870FDAB ... ",
		"Crawl Date": "6/June/2017"
	}
}

=== Metadata ===
metadata = {
	"date range start": "3/June/2017",
	"date range end": "6/June/2017",
	"locations": ["London", "London Bridge", "Borough Market"],
	"subjects": ["Arab States", "Qatar, "Youssef Zaghba Named", "Terror"]
}

This is your task:

=== Event ==='''


timeline_aspect_prompt_='''Follow my instructions as precisely as possible. Only provide the requested output, nothing more.

I am giving you the name of and event, an aspect of the event that you should focus on, and a JSON list of news articles about the event and its aspect.

Your task is to provide a timeline JSON of all the articles (from 1 to 10) that should focus on the event and its aspect. For each item in the timeline, provide references to the articles using their identifiers. Only use information that is contained in the articles.

This is an example:

=== Event and Aspect ===
Event: 2017 London Bridge attack
Aspect: Borough Market

=== News Articles ===
{
	"1": {
		"Title": "Sputnik International - Breaking News & Analysis - Radio, Photos, Videos, Infographics",
		"Snippet": "2017 Persian Gulf Disarray: Arab States Sever Relations With Qatar London Bridge and Borough Market ...  Software Vendors 13:17 Moroccan-Italian Youssef Zaghba Named as Third Perpetrator of London Attack 13 ... -word and Kathy Griffin&#8217;s Trump severed head mock-up are connected. 06 June 2017 Will London Attacks ...",
		"Crawl Date": "6/June/2017"
	},
	"2": {
		"Title": "Twilight Language: London Bridge Attack",
		"Snippet": "). Saturday, June 03, 2017 London Bridge Attack A white van mowed down pedestrians as it sped down London ...  Twitter Share to Facebook Share to Pinterest Labels: Borough Market , Jupiter , London Bridge ... Twilight Language: London Bridge Attack Twilight Language The twilight language explores hidden ... "},
		"Crawl Date": "2/August/2017"
	"10": {
		"Title": "World Bulletin / Europe",
		"Snippet": "-attacks Sun, 04 Jun 2017 10:28:05 GMT &acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market Europe ... -after-london-bridge-attack Mon, 05 Jun 2017 12:09:41 GMT No 'direct' proof Russia meddled in US ... &quot; title=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market&quot; alt=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike ... ",
		"Crawl Date": "8/June/2017"},
	"5": {
		"Title": "Scottish and Scots Diaspora News Feed",
		"Snippet": "-A3A116A60FDD Sun, 4 Jun 2017 11:03:56 -0400 London Bridge attack treated as terrorist incident Police have confirmed that incidents at London Bridge and nearby Borough Market are terrorist incidents  3870FDAB ... ",
		"Crawl Date": "6/June/2017"
	}
}

=== Timeline ===

[
	{
		"Date": "2017/June/3",
		"Text": "A white van mows down pedestrians on London Bridge and in Borough Market in a terrorist attack.",
		"Articles": ["1", "5"]
	},
	{
		"Date": "2017/June/4",
		"Text": "The London Bridge attack, including incidents at Borough Market, is officially confirmed as a terrorist incident by the police.",
		"Articles": ["2"]
	},
	{
		"Date": "2017/June/5",
		"Text": "Reports indicate that there is no 'direct' proof of Russia's involvement in the London Bridge and Borough Market attacks.",
		"Articles": ["10"]
	},
	{
		"Date": "2017/June/6",
		"Text": "Youssef Zaghba is identified as the third perpetrator of the London Bridge and Borough Market attack.",
		"Articles": ["1"]
	}
]

=== End of Timeline ===

This is your task:

=== Event ==='''

timeline_general_prompt_='''Follow my instructions as precisely as possible. Only provide the requested output, nothing more.

I am giving you the name of and event and a JSON list of news articles about the event.

Your task is to provide a timeline JSON of all the articles (from 1 to 10) that should focus on the event. You can use all articles. For each item in the timeline, provide references to the articles using their identifiers. Only use information that is contained in the articles.

This is an example:

=== Event and Aspect ===
Event: 2017 London Bridge attack

=== News Articles ===
{
	"1": {
		"Title": "Sputnik International - Breaking News & Analysis - Radio, Photos, Videos, Infographics",
		"Snippet": "2017 Persian Gulf Disarray: Arab States Sever Relations With Qatar London Bridge and Borough Market ...  Software Vendors 13:17 Moroccan-Italian Youssef Zaghba Named as Third Perpetrator of London Attack 13 ... -word and Kathy Griffin&#8217;s Trump severed head mock-up are connected. 06 June 2017 Will London Attacks ...",
		"Crawl Date": "6/June/2017"
	},
	"2": {
		"Title": "Twilight Language: London Bridge Attack",
		"Snippet": "). Saturday, June 03, 2017 London Bridge Attack A white van mowed down pedestrians as it sped down London ...  Twitter Share to Facebook Share to Pinterest Labels: Borough Market , Jupiter , London Bridge ... Twilight Language: London Bridge Attack Twilight Language The twilight language explores hidden ... "},
		"Crawl Date": "2/August/2017"
	"10": {
		"Title": "World Bulletin / Europe",
		"Snippet": "-attacks Sun, 04 Jun 2017 10:28:05 GMT &acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market Europe ... -after-london-bridge-attack Mon, 05 Jun 2017 12:09:41 GMT No 'direct' proof Russia meddled in US ... &quot; title=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike London Bridge, Borough Market&quot; alt=&quot;&acirc;&#128;&#152;Terror attacks&acirc;&#128;&#153; strike ... ",
		"Crawl Date": "8/June/2017"},
	"5": {
		"Title": "Scottish and Scots Diaspora News Feed",
		"Snippet": "-A3A116A60FDD Sun, 4 Jun 2017 11:03:56 -0400 London Bridge attack treated as terrorist incident Police have confirmed that incidents at London Bridge and nearby Borough Market are terrorist incidents  3870FDAB ... ",
		"Crawl Date": "6/June/2017"
	}
}

=== Timeline ===

[
	{
		"Date": "2017/June/3",
		"Text": "A white van mows down pedestrians on London Bridge and in Borough Market in a terrorist attack.",
		"Articles": ["1", "5"]
	},
	{
		"Date": "2017/June/4",
		"Text": "The London Bridge attack, including incidents at Borough Market, is officially confirmed as a terrorist incident by the police.",
		"Articles": ["5"]
	},
	{
		"Date": "2017/June/5",
		"Text": "Reports indicate that there is no 'direct' proof of Russia's involvement in the London Bridge and Borough Market attacks.",
		"Articles": ["10"]
	},
	{
		"Date": "2017/June/6",
		"Text": "Youssef Zaghba is identified as the third perpetrator of the London Bridge and Borough Market attack.",
		"Articles": ["1"]
	}
]

=== End of Timeline ===

This is your task:

=== Event ==='''



entities_df=pd.read_csv("~/web_archive_collections/arquivo.pt/all_events/expansion_terms/10_event_terms_types2.csv", sep="\t")
entities_df=entities_df.loc[entities_df["event_label"].notna(),]
events=list(entities_df["event_label"].unique())


chatgpt_results=pd.DataFrame(columns=["event","aspect","type", "prompt","answer"])
results_dict={}


def get_metadata_json(metadata_str):
    metadata_str = metadata_str.replace("metadata = ", "").replace("=== Metadata ===", "").strip()
    metadata_str = metadata_str[:-1].strip() if metadata_str.endswith(';') else metadata_str

    try:
        metadata_json = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        print("Cannot parse metadata string as JSON: ", metadata_str)
        return None
    return metadata_json


def get_json_timeline(timeline_str):
    timeline_str = timeline_str.replace("=== End of Timeline ===", "").replace("=== Timeline ===", "")
 
    timeline_str = timeline_str.strip()
 
    try:
        timeline_json = json.loads(timeline_str)
    except json.JSONDecodeError as e:
        print("Cannot parse timeline string as JSON: ", timeline_str)
        return None
    return timeline_json





def chatgpt(prompt_):
	response = openai.Completion.create(
  	engine="gpt-3.5-turbo-instruct",
    prompt=prompt_,
    
    max_tokens=600)
	return(response.choices[0].text.strip())



def timeline_fun(prompt_):
    timeline_str=chatgpt(prompt_)
    if "=== Timeline ===" in timeline_str:
        timeline_general_number=timeline_str.find("=== Timeline ===")
        timeline_str=timeline_str[timeline_general_number:]
    
    timeline_str = timeline_str.replace("=== End of Timeline ===", "").replace("=== Timeline ===", "")
 
    timeline_str = timeline_str.strip()
 
    try:
        timeline_json = json.loads(timeline_str)
        return timeline_json
    except json.JSONDecodeError as e:
        print("Cannot parse timeline string as JSON: ", timeline_str)
        return timeline_fun(prompt_)
     #   return None
    #return timeline_json

def timeline_fun_reduced(prompt_):
    timeline_str=chatgpt_reduced(prompt_)
    if "=== Timeline ===" in timeline_str:
        timeline_general_number=timeline_str.find("=== Timeline ===")
        timeline_str=timeline_str[timeline_general_number:]
    
    timeline_str = timeline_str.replace("=== End of Timeline ===", "").replace("=== Timeline ===", "")
 
    timeline_str = timeline_str.strip()
 
    try:
        timeline_json = json.loads(timeline_str)
        return timeline_json
    except json.JSONDecodeError as e:
        print("Cannot parse timeline string as JSON: ", timeline_str)
        return timeline_fun_reduced(prompt_)
 


def metadata_fun(prompt_):
    metadata_str=chatgpt(prompt_)
    if "=== Metadata ===" in metadata_str:
        metadata_number=metadata_str.find("=== Metadata ===")
        metadata_str=metadata_str[metadata_number:]
    
    metadata_str = metadata_str.replace("metadata = ", "").replace("=== Metadata ===", "").strip()
    metadata_str = metadata_str[:-1].strip() if metadata_str.endswith(';') else metadata_str

    try:
        metadata_json = json.loads(metadata_str)
        return metadata_json
    except json.JSONDecodeError as e:
        print("Cannot parse metadata string as JSON: ", metadata_str)
        return metadata_fun(prompt_)



    timeline_str = timeline_str.replace("=== End of Timeline ===", "").replace("=== Timeline ===", "")
 
    timeline_str = timeline_str.strip()
 
    try:
        timeline_json = json.loads(timeline_str)
        return timeline_json
    except json.JSONDecodeError as e:
        print("Cannot parse metadata string as JSON: ", timeline_str)
        return timeline_fun(prompt_)
 



def summary_fun(prompt_):
    output=chatgpt(prompt_)
    if "=== Summary ===" in output:
        return output
    else:
        print("summary didn't work")
        return summary_fun(prompt_)




def get_json(df):
    tmp_dict={}
    df=df.reset_index(drop=True)
    for i in range(df.shape[0]):
        tmp_dict[str(i+1)]={}
        tmp_dict[str(i+1)]["Title"]=df.iloc[i]["title"]
        tmp_dict[str(i+1)]["Snippet"]=df.iloc[i]["cleaned_snippet"]
        tmp_dict[str(i+1)]["Crawl Date"]=df.iloc[i]["date"]
    return(tmp_dict)





summary_df=pd.DataFrame(columns=["event","aspect","summary"])
for event in events:
    
    results_dict[event]={}
    title=event.replace("_"," ")
    entities_tmp=entities_df.loc[entities_df["event_label"]==event,]
    entities=list(entities_tmp["term"].unique())
    try:
        entities.remove("Middlesex")
    except:
        print("Not in the list")
    #aspects=entities+["when","cause","result","NO-ASPECT"]
    if "Election" in event or "election" in event or "prix" in event or "Prix" in event or "song" in event or "Song" in event or "Film" in event or "film" in event:
        aspects=entities+["when","result","NO-ASPECT"]
    else:
        aspects=entities+["when","result","NO-ASPECT","cause"]

    aspects_=[x.title() for x in aspects]
    final_dict={}
    final_dict["Event_Names"]=title
    final_dict[title]={}
    final_dict[title]["Aspects"]=aspects_
    
    all_data=pd.DataFrame()
    for aspect in aspects:
        try:
            results_dict[event][aspect]={}
            tmp_metadata_dict={}
            print("*******************************")
            print("aspect is", aspect)
            if aspect=="NO-ASPECT":
                address="~/web_archive_collections/arquivo.pt/all_events/"+event+"/event_diversified_scores.csv"
            else:
                address="~/web_archive_collections/arquivo.pt/all_events/"+event+"/"+aspect+"_diversified_scores.csv"
            final_dict[title][aspect]={}
            data=pd.read_csv(address, sep=",")
            data["title"]=data["title"].fillna("")
            all_data=all_data.append(data)
            data["rank"]=data["final_score2"].rank(method="first", ascending=False)
            data=data.sort_values(by="rank", ascending=True)
            data=data.reset_index(drop=True)

            if aspect in ["cause","when","result","NO-ASPECT"]:

                upper_aspect="no upper aspect"
            else:
                upper_aspect=list(entities_tmp.loc[entities_tmp["term"]==aspect,"type"])[0]
            final_dict[title][aspect]["upper_aspect"]=upper_aspect
            final_dict[title][aspect]["Document_Ranking"]={}
            final_dict[title][aspect]["Document_Ranking"]["Documents_List"]=list(data.head(20)["title"])
            final_dict[title][aspect]["Overview"]={}
            final_dict[title][aspect]["Timeline"]={}
            for i in range(min(100,data.shape[0]-1)):
                final_dict[title][aspect]["Document_Ranking"][i+1]={}
                final_dict[title][aspect]["Document_Ranking"][i+1]["Title"]=data.iloc[i]["title"]
                final_dict[title][aspect]["Document_Ranking"][i+1]["Snippet"]=data.iloc[i]["cleaned_snippet"]
                final_dict[title][aspect]["Document_Ranking"][i+1]["Web Archive URL"]=data.iloc[i]["archive_url"]
                final_dict[title][aspect]["Document_Ranking"][i+1]["Live Web URL"]=data.iloc[i]["original_url"]
                final_dict[title][aspect]["Document_Ranking"][i+1]["Date"]=data.iloc[i]["date"]
            json_file=get_json(data.head(10))
            json_file=json.dumps(json_file)
            timeline_aspect_prompt=timeline_aspect_prompt_+"\n\nEvent: "+event+"\nAspect: "+aspect+"\n\n=== News Articles ===\n\n"+json_file+"\n\n=== Timeline ==="
            metadata_prompt=metadata_prompt_+"\n\nEvent: "+event+"\n\n=== News Articles ===\n\n"+json_file+"\n"
            summary_aspect_prompt=summary_aspect_prompt_+"\n\nEvent: "+event+"\nAspect: "+aspect+"\n\n=== News Articles ===\n\n"+json_file+"\n\n=== Summary ==="
            timeline_answer=chatgpt(timeline_aspect_prompt[:min(len(timeline_aspect_prompt.split(" ")),4097)])
            if "=== Timeline ===" in timeline_answer:
                timeline_number=timeline_answer.find("=== Timeline ===")
                timeline_answer=timeline_answer[timeline_number:]
            timeline_answer=get_json_timeline(timeline_answer)
            
            
            metadata_answer=metadata_fun(metadata_prompt[:min(len(metadata_prompt.split(" ")),4097)])
            if "=== Metadata ===" in metadata_answer:
                metadata_number=metadata_answer.find("=== Metadata ===")
                metadata_answer=metadata_answer[metadata_number:]
            metadata_answer=get_metadata_json(metadata_answer)


           
            
            if "=== Summary ===" in summary_answer:
                summary_number=summary_answer.find("=== Summary ===")
                summary_answer=summary_answer[summary_number:]

 
            results_dict[event][aspect]["timeline"]=timeline_answer
            results_dict[event][aspect]["metadata"]=metadata_answer
            results_dict[event][aspect]["summary"]=summary_answer
          
            
            

        except:
            print("Exception for: ", event, aspect)

    all_data["rank"]=all_data.groupby(["title","content","date","original_url","archive_url"])["final_score2"].rank("first", ascending=False)
    all_data=all_data.loc[all_data["rank"]==1,]
        #all_data=all_data[["title","snippet","date","original_url","archive_url","final_score3"]].drop_duplicates()
    all_data["final_rank"]=all_data["final_score2"].rank(method="first", ascending=False)
    all_data=all_data.sort_values(by="final_rank", ascending=True)
    all_data=all_data.reset_index(drop=True)
    all_data=all_data.loc[:100,]
    final_dict[title]["Global_Ranking"]={}
    final_dict[title]["Global_Ranking"]["Documents_List"]=list(all_data["title"])

    for j in range(all_data.shape[0]):
            final_dict[title]["Global_Ranking"][j+1]={}
            final_dict[title]["Global_Ranking"][j+1]["Title"]=all_data.iloc[j]["title"]
            final_dict[title]["Global_Ranking"][j+1]["Snippet"]=all_data.iloc[j]["cleaned_snippet"]
            final_dict[title]["Global_Ranking"][j+1]["Web Archive URL"]=all_data.iloc[j]["archive_url"]
            final_dict[title]["Global_Ranking"][j+1]["Live Web URL"]=all_data.iloc[j]["original_url"]
            final_dict[title]["Global_Ranking"][j+1]["Date"]=all_data.iloc[j]["date"]
    
    final_dict[title]["Global_Ranking"]["Overview"]={}
    final_dict[title]["Global_Ranking"]["Timeline"]={}
    json_file=get_json(all_data.head(20))
    json_file=json.dumps(json_file)
    
    timeline_general_prompt=timeline_general_prompt_+"\n\nEvent: "+event+"\n=== News Articles ===\n\n"+json_file+"\n\n=== Timeline ==="
    metadata_prompt=metadata_prompt_+"\n\nEvent: "+event+"\n\n=== News Articles ===\n\n"+json_file+"\n"
    summary_general_prompt=summary_general_prompt_+"\n\nEvent: "+event+"\n=== News Articles ===\n\n"+json_file+"\n\n=== Summary ==="
 
 
    try:
        timeline_general_answer=timeline_fun(timeline_general_prompt)           
    except: 
        timeline_general_answer=timeline_fun_reduced(timeline_general_prompt)       
    metadata_general_answer=metadata_fun(metadata_prompt)
                
    metadata_general_answer=chatgpt(metadata_prompt)
    if "=== Metadata ===" in metadata_general_answer:
        metadata_general_number=metadata_general_answer.find("=== Metadata ===")
        metadata_general_answer=metadata_general_answer[metadata_general_number:]
    metadata_general_answer=get_metadata_json(metadata_general_answer)
            
    summary_general_answer=chatgpt(summary_general_prompt)
    if "=== Summary ===" in summary_general_answer:
        summary_general_number=summary_general_answer.find("=== Summary ===")
        summary_general_answer=summary_general_answer[summary_general_number:]

    results_dict[event]["general"]={}
    results_dict[event]["general"]["timeline"]=timeline_general_answer
    results_dict[event]["general"]["metadata"]=metadata_general_answer
    results_dict[event]["general"]["summary"]=summary_general_answer
         

