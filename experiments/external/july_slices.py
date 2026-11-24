"""Post-lock diagnostics, not gates: slices defined before unblinding.
No product, organization name or case-level text in the public tables.
"""
import argparse, json, hashlib
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy.stats import norm

def hashfile(path):
 with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def wilson(k,n):
 if not n:return None,None
 z=norm.ppf(.975);p=k/n;den=1+z*z/n
 center=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return max(0.,center-h),min(1.,center+h)
def report(repo, policy_dir, lock, blind, out):
 sys.path.insert(0,str(policy_dir));from shift_policy import apply
 policy=json.loads(lock.read_text());x=np.load(blind/'july_predictions.npz',allow_pickle=False)
 truth=np.load(blind/'july_labels.npz',allow_pickle=False);assert np.array_equal(x['id'],truth['id'])
 y=truth['y'];_,dest,auto=apply(x['p_legacy'],policy);dest=np.asarray(dest);auto=np.asarray(auto).astype(bool)
 assert y.shape==dest.shape==auto.shape
 earlier=pd.read_pickle(repo/'data'/'cohort.pkl')
 train=earlier[earlier.split.eq('train')]
 known=train.company.dropna().unique()
 wc=x['word_count'];dates=pd.DatetimeIndex(x['date']);company=x['company']
 familiar=np.isin(company,known)
 labels=[('institution in historical training',np.where(familiar,'Seen','Unseen or unspecified')),
         ('published narrative length in words',np.select([wc<50,wc<200],['10–49','50–199'],'200+')),
         ('receipt date band',np.where(dates.day<=7,'Jul 01–07',np.where(dates.day<=14,'Jul 08–14',np.where(dates.day<=21,'Jul 15–21','Jul 22–29'))))]
 wrong=auto&(dest!=y)
 rows=[]
 for attribute,membership in labels:
  for group in np.unique(membership):
   m=membership==group;n=int(m.sum());sent=int(auto[m].sum());err=int(wrong[m].sum());lo,hi=wilson(err,sent)
   rows.append({'attribute':attribute,'subgroup':group,'eligible':n,'auto':sent,'wrong':err,'coverage':sent/n,'risk':err/sent if sent else None,'risk_lo95':lo,'risk_hi95':hi})
 # Resample lender clusters as entire units. Uniform cluster bootstrap weights
 # cases according to multiplicity of their provider, resulting in a case-weighted estimand.
 ag=pd.DataFrame({'company':company,'cases':np.ones(len(y),int),'auto':auto.astype(int),'wrong':wrong.astype(int)}).groupby('company',sort=True)[['cases','auto','wrong']].sum()
 counts=ag.to_numpy();rng=np.random.default_rng(15071);s=[]
 for i in range(2500):
  vals=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0)
  if vals[1]:s.append([vals[1]/vals[0],vals[2]/vals[1]])
 out.mkdir(exist_ok=True,parents=True)
 pd.DataFrame(rows).to_csv(out/'july_audit_slices.csv',index=False)
 boot={'n_companies':len(ag),'bootstrap_draws':len(s),'method':'Resample full institution clusters uniformly with replacement including missing tenant as its own cluster; percentile 95% intervals; diagnostics, not new gates.','coverage_interval':np.quantile(np.array(s)[:,0],[.025,.975]).tolist(),'risk_interval':np.quantile(np.array(s)[:,1],[.025,.975]).tolist(),'locked_policy_sha256':hashfile(lock),'all_July_n':len(y),'all_July_autorouted':int(auto.sum()),'all_July_wrong':int(wrong.sum())}
 (out/'july_institution_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
 print('Saved aggregate audit diagnostics only.')
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for a in ['repo','policy-dir','lock','blind','out']:p.add_argument('--'+a,required=True,type=Path)
 a=p.parse_args();report(a.repo.resolve(),a.policy_dir.resolve(),a.lock.resolve(),a.blind.resolve(),a.out.resolve())
