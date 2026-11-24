"""Coverage of disclosed narrative by product group. No text leaves the aggregate analysis."""
from pathlib import Path
import pandas as pd, json, zipfile
import sys
root=Path(__file__).resolve().parent
def summarize(archives, output):
 rows=[]
 for period,path in archives:
  totals={}
  with zipfile.ZipFile(path) as z:
   for df in pd.read_csv(z.open(z.namelist()[0]),usecols=['Product','Date received','Consumer complaint narrative'],dtype='string',chunksize=50000):
    if period=='Historical test':df=df[df['Date received'].between('2020-09-01','2021-04-30')]
    group=df['Product'].str.lower().str.startswith('credit report').fillna(False)
    eligible=df['Consumer complaint narrative'].fillna('').str.count(r'\S+').ge(10)
    x=pd.crosstab(group,eligible)
    for k in (False,True):
     for l in (False,True):totals[k,l]=totals.get((k,l),0)+int(x.reindex(index=[False,True],columns=[False,True],fill_value=0).loc[k,l])
  for k in (False,True):rows.append(dict(period=period,group='Credit reports' if k else 'All other products',public_records=totals[k,0]+totals[k,1],usable_narrative=totals[k,1]))
 df=pd.DataFrame(rows);df['narrative_rate']=df.usable_narrative/df.public_records
 df['share_of_public_records']=df.public_records/df.groupby('period').public_records.transform('sum')
 df['share_of_usable_texts']=df.usable_narrative/df.groupby('period').usable_narrative.transform('sum')
 df.to_csv(output,index=False);return df
if __name__=='__main__':
 import argparse
 q=argparse.ArgumentParser();q.add_argument('historical_zip');q.add_argument('june_zip');q.add_argument('output');a=q.parse_args()
 print(summarize([('Historical test',a.historical_zip),('June 2026',a.june_zip)],a.output).to_string(index=False))
