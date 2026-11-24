"""Local shadow-study randomization, outcome collection schema and fixed-look power.
Use synthetic or locally pseudonymized case references only. No raw narrative is needed.
Usage: python experiments/transfer/pilot.py design | assign strata.csv assignments.csv --seed 421 |
       analyze scored.csv; see docs/field-protocol.md before using this with humans.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta,binom
DESK_N=400; TARGET=.05; FAMILY_ALPHA=.05; DESKS=['Cards','Mortgages']
def design():
 alpha=FAMILY_ALPHA/len(DESKS)
 def upper(k):return float(beta.ppf(1-alpha,k+1,DESK_N-k)) if k<DESK_N else 1.
 mx=max(k for k in range(DESK_N) if upper(k)<TARGET)
 return dict(n_per_destination=DESK_N,maximum_wrong=mx,next_fails=mx+1,upper_if_maximum=upper(mx),upper_if_next=upper(mx+1),single_look_alpha_per_desk=alpha,pass_probability_at_2pct=float(binom.cdf(mx,DESK_N,.02)),pass_probability_at_3pct=float(binom.cdf(mx,DESK_N,.03)),assumptions='One fixed look per queue; independent Bernoulli opportunities. Template, reviewer, client or date clustering can reduce effective sample size.')
def assign(rows,seed):
 if len({r['case_key'] for r in rows})!=len(rows):raise ValueError('Each case appears once.')
 # Paired block randomization: pairs within desk/short flag. Unpaired last case gets a fair coin.
 rng=np.random.default_rng(seed);out=[];strata=defaultdict(list)
 for r in rows:
  if not r['case_key'] or r['length_band'] not in ('short','long'):raise ValueError('Invalid case metadata.')
  strata[(r['suggested_desk'],r['length_band'])].append(r)
 for key,rs in sorted(strata.items()):
  orig=sorted(rs,key=lambda r:r['case_key']);order=rng.permutation(len(orig))
  for k in range(0,len(order),2):
   v=int(rng.integers(2))
   for j,ix in enumerate(order[k:k+2]):
    r=orig[ix]
    out.append({k:r[k] for k in ['case_key','suggested_desk','length_band']}|{'pair_id':f'{key[0]}:{key[1]}:{k//2}','arm':('normal','shortlist')[(v+j)%2]})
 return sorted(out,key=lambda r:r['case_key'])
def observed(rows):
 if len({r['case_key'] for r in rows})!=len(rows):raise ValueError('Duplicate case key.')
 pairs=defaultdict(list);armrows=defaultdict(list)
 for r in rows:
  if r['arm'] not in ('normal','shortlist'):raise ValueError('Invalid arm')
  if not r['adjudicated_queue'] or not r['decision']:raise ValueError('Missing reviewed outcomes; do not silently exclude a case')
  sec=float(r['active_seconds'])
  if not np.isfinite(sec) or sec<=0:raise ValueError('Active seconds must be positive')
  # Decision includes an explicit other/review option.
  out=(r['arm'],float(r['decision']==r['adjudicated_queue']),sec)
  pairs[r['pair_id']].append(out);armrows[r['arm']].append(out[1:])
 if set(armrows)!={'normal','shortlist'}:raise ValueError('Both arms are needed')
 dif=[]
 for group in pairs.values():
  if len(group)==1:continue
  if len(group)!=2 or set(r[0] for r in group)!={'normal','shortlist'}:raise ValueError('Broken matched block')
  m={r[0]:np.array(r[1:]) for r in group};dif.append(m['shortlist']-m['normal'])
 ret={'arms':{a:{'n':len(v),'accuracy':float(np.mean(v,axis=0)[0]),'mean_active_seconds':float(np.mean(v,axis=0)[1])} for a,v in armrows.items()},'complete_pairs':len(dif),'singleton_cases':len(rows)-2*len(dif)}
 if len(dif)>=2:
  arr=np.array(dif);rng=np.random.default_rng(15071);boot=np.array([arr[rng.integers(0,len(arr),len(arr))].mean(axis=0) for _ in range(5000)])
  ret['paired_differences_hint_minus_normal']={k:{'point':float(arr.mean(axis=0)[j]),'bootstrap95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,k in enumerate(['accuracy_fraction','mean_active_seconds'])}
 return ret
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);sub.add_parser('design')
 a=sub.add_parser('assign');a.add_argument('input',type=Path);a.add_argument('output',type=Path);a.add_argument('--seed',type=int,required=True)
 a=sub.add_parser('analyze');a.add_argument('input',type=Path)
 arg=p.parse_args()
 if arg.cmd=='design':print(json.dumps(design(),indent=2));return
 with arg.input.open() as f:rows=list(csv.DictReader(f))
 if arg.cmd=='analyze':print(json.dumps(observed(rows),indent=2));return
 if not rows:raise ValueError('No cases')
 new=assign(rows,arg.seed)
 with arg.output.open('x') as f:
  wr=csv.DictWriter(f,fieldnames=list(new[0]));wr.writeheader();wr.writerows(new)
if __name__=='__main__':main()
