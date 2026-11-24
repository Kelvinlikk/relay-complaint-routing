"""Descriptive, never a deployed rule: carry estimated 2020/21 within-product
routing patterns into the June 2026 observed product mix. No July labels read.
Usage: python mix_replay.py --results-dir <baseline results> --out <folder>
"""
from pathlib import Path
import argparse,json
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('.'));args=p.parse_args()
 q=pd.read_csv(args.results_dir/'queue_metrics.csv');cm=pd.read_csv(args.results_dir/'auto_confusion.csv')
 old=q[q.split=='test'].set_index('queue').true;new=q[q.split=='shift'].set_index('queue').true
 w=(new/new.sum())/(old/old.sum())
 m=cm[cm.split=='test'].copy();m['reweighted']=m.n*m.actual.map(w); rows=[]
 def calc(df,field,period,den):
  for name,sub in [('all',df),('placed in Credit reports',df[df.route=='Credit reports'])]:
   n=float(sub[field].sum()); wrong=float(sub.loc[sub.actual!=sub.route,field].sum())
   rows.append({'period':period,'destination_scope':name,'coverage':n/den,'wrong_among_routed':wrong/n})
 calc(m,'n','observed primary',old.sum());calc(m,'reweighted','old conditional behavior, June true-label proportions',old.sum());calc(cm[cm.split=='shift'],'n','observed June',new.sum())
 result=pd.DataFrame(rows);result.to_csv(args.out/'mix_replay.csv',index=False);print(result.to_string(index=False))
if __name__=='__main__':main()
