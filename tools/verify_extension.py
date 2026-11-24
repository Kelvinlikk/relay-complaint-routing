"""Independent acceptance checks on the extended deliverable; no consumer data."""
from pathlib import Path
import csv,json
from math import isclose
def verify(root, trial=True):
 base=root/'experiments/transfer/results'
 def tab(n):return list(csv.DictReader((base/(n+'.csv')).open()))
 a=json.loads((base/'source_audit.json').read_text()); assert a['august']['records']==667140 and a['august']['eligible']==a['august']['retained']==1
 assert a['may']['sampled']-sum(a['may']['removed'].values())==a['may']['retained']==5254
 g={(r['period'],r['policy'],r['group']):r for r in tab('gates')}
 for period,tot,count0,err0,count1,err1 in [('May',5254,479,13,461,12),('July',8830,928,17,899,12),('August',1,0,0,0,0)]:
  for nm,n,w in [('June rule',count0,err0),('June rule + length hold',count1,err1)]:
   r=g[period,nm,'All']; assert [int(r[k]) for k in ['group_n','routed','wrong']]==[tot,n,w]
   if n:assert float(r['ci_low'])<=w/n<=float(r['ci_high'])
   else:assert r['wrong_per_route']==r['ci_low']==r['ci_high']==''
   assert sum(int(g[period,nm,q]['wrong']) for q in ['Cards','Mortgages'])==w
 z=tab('reviewer_shortlist');s=tab('shortlist_by_product');pdiff=tab('paired_shortlist')
 for r in z:
  if r['k']!='2':continue
  period,method=r['period'],r['method'];n=int(r['n']);num=int(r['correct_in_list'])
  group=[v for v in s if (v['method'],v['period'])==(method,period)]
  assert n==sum(int(v['n']) for v in group) and num==sum(int(v['correct_in_top2']) for v in group)
  assert n==int(g[period,'June rule + length hold','All']['group_n'])-int(g[period,'June rule + length hold','All']['routed'])
  for c in pdiff:
   if c['period']==period and c['versus']==method:
    model=next(v for v in z if v['period']==period and v['method']=='Model' and v['k']=='2')
    assert int(c['model_only'])-int(c['baseline_only'])==int(model['correct_in_list'])-num
    assert float(c['day_cluster_lower'])<float(c['paired_lift'])<float(c['day_cluster_upper'])
 if not trial:return
 from experiments.transfer.pilot import assign,observed,design
 test=design();assert test['maximum_wrong']==11 and test['upper_if_maximum']<.05<test['upper_if_next']
 # Synthetic, known-effect paired trial: completely separate from model performance and public cases.
 sample=[dict(case_key=f'SYN{i}',suggested_desk='Cards',length_band='long') for i in range(8)]
 trial=assign(sample,19)
 for r in trial:
  assisted=r['arm']=='shortlist';r.update(adjudicated_queue='Cards',decision='Cards' if assisted else 'Accounts',active_seconds=45 if assisted else 60)
 output=observed(trial);diff=output['paired_differences_hint_minus_normal']
 assert output['complete_pairs']==4 and diff['accuracy_fraction']['point']==1 and diff['mean_active_seconds']['point']==-15
 print('Verified aggregate rates, grouped ranks, paired contrasts, release boundary and synthetic trial analysis.')
if __name__=='__main__':
 import sys
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));verify(root)
