import pandas as pd
from langdetect import detect

entities_df=pd.read_csv("./data/event_aspect_terms.tsv", sep="\t")
event_ids=pd.read_csv("./data/event_ids.tsv", sep="\t")
event_ids=event_ids.rename(columns={"eventkg_id":"event", "label":"event_label"})
entities_df=pd.merge(left=entities_df, right=event_ids, how="left", on="event")
events=["Great Fire of London"]
for event in events:
    entities_tmp=entities_df.loc[entities_df["event_label"]==event,]
    entities=list(entities_tmp["term"].unique())
    entities=entities+["when","cause","result","event"]
    for entity in entities:
        try:
            data=pd.read_csv("./data/"+event.lower().replace(" ","_")+"/"+entity+".csv",sep=",")
            data=data.loc[10:,]
            data.columns=data.iloc[0]
            data=data[1:].reset_index(drop=True)
            data["Snippet"]=data["Snippet"].str.replace("<em>","")
            data["Snippet"]=data["Snippet"].str.replace("</em>","")
            data["Snippet"]=data["Snippet"].str.replace("</span>","")
            data["Snippet"]=data["Snippet"].str.replace('<span class="ellipsis">',"")
            data1=data[["Year","Month","Day","Title","Snippet","Link to archive"]]
            
            data=data.loc[data["Snippet"].notna(),]
            try:
                data["lang"]=data.apply(lambda row: detect(row.Snippet), axis=1)
                data=data.loc[data["lang"]=="en"]
                data.to_csv("/data/events/"+event+"/"+entity+"_results.tsv", sep="\t")
            except:
                data["lang"]=""
                for i in range(data.shape[0]):
                    try:
                        lang=detect(data.iloc[i]["Snippet"])
                        data.loc[i,"lang"]=lang
                    except:
                        continue
                data=data.loc[data["lang"]=="en"]
                data.to_csv("./data/"+event.lower().replace(" ","_")+"/"+entity+"_results.tsv", sep="\t")

        except:
            print("no data for: ", event, entity)

