"""Decision economics per 10,000 eligible cases vs manual-first. No actual labor savings assumed."""
from pathlib import Path
from math import ceil
import csv,argparse
def scenarios(n,routed,wrong):
 for label,rm,cost in [('High rework',2,50),('Base case',4,25),('Long review',7,10)]:
  for audit in [0,.1,.25]:
   cost_review=rm*30/60
   labor=routed*cost_review
   audit_cost=routed*audit*cost_review
   rework=wrong*cost
   yield dict(scenario=label,review_minutes=rm,wage=30,wrong_route_cost=cost,audit_rate=audit,n=n,routed=routed,wrong=wrong,first_pass_hours_per_10000=(1-audit)*routed*rm/60*10000/n,review_value_per_10000=labor*10000/n,audit_cost_per_10000=audit_cost*10000/n,rework_per_10000=rework*10000/n,net_per_10000=(labor-audit_cost-rework)*10000/n,break_even_wrong_rate=(1-audit)*cost_review/cost)
def main():
 q=argparse.ArgumentParser();q.add_argument('total',type=int);q.add_argument('routed',type=int);q.add_argument('wrong',type=int);q.add_argument('output',type=Path);a=q.parse_args()
 if not 0<=a.wrong<=a.routed<=a.total or a.total<=0:raise ValueError('Unexpected counts')
 rows=list(scenarios(a.total,a.routed,a.wrong))
 with a.output.open('w') as f:
  wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
 print(next(v for v in rows if v['scenario']=='Base case' and v['audit_rate']==.1))
if __name__=='__main__':main()
