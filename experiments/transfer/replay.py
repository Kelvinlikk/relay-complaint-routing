"""Frozen destination gates and ranked assistance. Only aggregates leave the experiment.
From repo root after original src/prepare.py and July --stage prepare:
python experiments/transfer/replay.py --fetch
No torch is required because the frozen serving ensemble gives TinyBERT zero weight.
"""
from pathlib import Path
import argparse, csv, gc, hashlib, json, re, sys, zipfile
import numpy as np, pandas as pd, joblib, requests
from collections import Counter
from catboost import CatBoostClassifier
from datasketch import MinHash,MinHashLSH
from scipy.special import softmax
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent; DATA=ROOT/'data'; OUT=HERE/'results'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'/'external'),str(ROOT/'experiments'/'policy')]
from prepare import LABELS,clean,label,shingles
from july_holdout import mkhash
from shift_policy import apply,ci
URL='https://files.consumerfinance.gov/f/documents/'
FILES={'may':('CCDB_Export_18_May_2026.zip','311fe537afac63230d52f41529bb91361ce5a9844d7450866538bb0456bceade'),
       'june':('CCDB_Export_19_June_2026.zip','3021c8af9eeed4971a92f15be21c7ce08cf3c1337151de255e5b12bf782049df'),
       'july':('CCDB_Export_20_July_2026.zip','aafebbaaf0962c3d29532411ac2133c01bae830b7ea6ed1931a6e42278437d18'),
       'august':('CCDB_Export_21_August_2026.zip','c64b8147b499d6236d7d591bfdbe8db45af80694a1afb6bd6a739f2789b8b2b1')}
PRIV=DATA/'transfer_replay';PRIV.mkdir(parents=True,exist_ok=True)
KW=json.loads((HERE/'keywords.json').read_text())
POLICY=json.loads((ROOT/'experiments/policy/locked_policy.json').read_text())
DATEKEY={'may':'2026-05','june':'2026-06','july':'2026-07','august':'2026-08'}
COLS={'Date received':'date','Product':'product','Consumer complaint narrative':'text','Company':'company','State':'state','Complaint ID':'id'}
def dump(o,p):
 p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def get(m,fetch):
 fn,h=FILES[m];p=(DATA/'blind_july' if m=='july' else DATA)/fn
 if not p.exists() and fetch:
  p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.download')
  with requests.get(URL+fn,stream=True,timeout=150) as r:
   r.raise_for_status()
   with tmp.open('wb') as f:
    for b in r.iter_content(1<<20): f.write(b)
  tmp.replace(p)
 if not p.is_file():raise FileNotFoundError(f'Missing {p}; use --fetch or put the exact archive there.')
 with p.open('rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=h: raise ValueError('Changed source '+m)
 return p
def read_month(m,fetch):
 rows=[];total=0;any_text=0
 with zipfile.ZipFile(get(m,fetch)) as z:
  member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
  for d in pd.read_csv(z.open(member),usecols=list(COLS),chunksize=35000,low_memory=False):
   d=d.rename(columns=COLS);total+=len(d)
   if not d.date.str.startswith(DATEKEY[m]).all():raise ValueError('Unexpected date range: '+m)
   d=d[d.text.notna()].copy();sz=d.text.astype(str).str.split().str.len();any_text+=int((sz>0).sum());d['words']=sz
   rows.append(d[sz>=10])
 g=pd.concat(rows,ignore_index=True).sort_values(['date','id']).reset_index(drop=True);assert g.id.is_unique
 return g,dict(records=total,any_text=any_text,eligible=len(g),min_words=10)
def dedup(periods,prior):
 # Index new candidates, query backwards against the reference calendar. Jaccard and LSH band matches
 # are symmetric. This avoids holding more than 60k large reference shingle sets simultaneously.
 perm=MinHash(num_perm=64,seed=15071).permutations
 target={}; sigs={}; hits={}; exact_map={}; lsh=MinHashLSH(threshold=.73,num_perm=64)
 for m,d in periods.items():
  for r in d.itertuples():
   k=(m,r.Index);s=clean(r.text);v=shingles(s);h=hashlib.sha256(s.encode()).digest()
   sigs[k]=h;target[k]=(v,mkhash(v,perm));exact_map.setdefault(h,[]).append(k);lsh.insert(k,target[k][1])
 seen=set()
 for r in prior.itertuples(index=False):
  s=clean(r.text);h=hashlib.sha256(s.encode()).digest()
  if h in seen:continue
  seen.add(h)
  for k in exact_map.get(h,[]):hits[k]='exact_prior'
  v=shingles(s)
  for k in lsh.query(mkhash(v,perm)):
   if k in hits:continue
   w=target[k][0]
   if len(v&w)>=.80*len(v|w): hits[k]='near_prior'
 result={};summ={}
 for m,d in periods.items():
  seen_m=set();kept=[];within=MinHashLSH(threshold=.73,num_perm=64);loss=Counter()
  for i in range(len(d)):
   k=(m,i);h=sigs[k];v,mh=target[k]
   if k in hits: loss[hits[k]]+=1;continue
   if h in seen_m: loss['exact_within']+=1;continue
   if any(len(v & target[(m,j)][0])>=.80*len(v|target[(m,j)][0]) for j in within.query(mh)):
    loss['near_within']+=1;continue
   seen_m.add(h);within.insert(i,mh);kept.append(i)
  result[m]=d.iloc[kept].copy().reset_index(drop=True)
  summ[m]=dict(removed=dict(sorted(loss.items())),retained=len(kept),retained_id_sha256=hashlib.sha256(','.join(result[m].id.astype(str)).encode()).hexdigest())
 return result,summ
def score(d):
 model=joblib.load(ROOT/'models/linear.joblib');meta=CatBoostClassifier();meta.load_model(str(ROOT/'models/tab.cbm'))
 base=json.loads((ROOT/'results/policy.json').read_text());t=base['temperatures'];assert base['weights']=={'TF-IDF':.8,'TinyBERT':0.,'CatBoost':.2}
 parts=d[['company','state']].fillna('[unknown]').astype(str)
 def temp(x,n):return softmax(np.log(np.clip(x,1e-9,1))/n,axis=1)
 pt=model.predict_proba(d.text);pc=meta.predict_proba(parts,thread_count=2)
 old=temp(.8*temp(pt,t['TF-IDF'])+.2*temp(pc,t['CatBoost']),t['Combined'])
 return apply(old,POLICY)
def keyword_order(text,priors):
 # Each distinct substring is one vote per queue. Ties go to June frequency, then label order.
 state=' '.join(str(text).casefold().split());strength=[sum(s in state for s in KW[l]) for l in LABELS]
 return sorted(range(len(LABELS)),key=lambda j:(-strength[j],-priors[j],j))
def bootstrap(days,delta,seed=15071):
 codes,idx=np.unique(days,return_inverse=True);n=np.bincount(idx);v=np.bincount(idx,weights=delta);rng=np.random.default_rng(seed)
 arr=rng.integers(0,len(codes),size=(5000,len(codes)));q=v[arr].sum(axis=1)/n[arr].sum(axis=1)
 return [float(v) for v in np.quantile(q,[.025,.975])]
def summarize(tag,d,y,priors,transfer,shortlists,byq,paired):
 if len(d)==0:return
 prob,pred,gate=score(d);known=d.company.fillna('').astype(str).str.strip().ne('').to_numpy()
 gate=gate&known;words=d.words.to_numpy();ranks=np.argsort(-prob,axis=1,kind='stable')
 longgate=gate&(words>=50)
 for nm,g in [('June rule',gate),('June rule + length hold',longgate)]:
  for lab in ['All','Cards','Mortgages','10–49 words']:
   group=np.ones(len(y),bool) if lab=='All' else ((words<50) if lab=='10–49 words' else pred==LABELS.index(lab))
   auto=g&group;n=int(auto.sum());w=int((auto&(pred!=y)).sum());den=int(group.sum())
   transfer.append(dict(period=tag,policy=nm,group=lab,group_n=den,routed=n,wrong=w,share_of_group=n/den if den else None,share_ci_low=ci(n,den)[0],share_ci_high=ci(n,den)[1],wrong_per_route=w/n if n else None,ci_low=ci(w,n)[0],ci_high=ci(w,n)[1]))
 if tag=='August' and len(d)<100: return
 pop_order=np.argsort(-np.array(priors),kind='stable')
 held=~longgate;ii=np.flatnonzero(held)
 keyword=np.array([keyword_order(d.text.iloc[i],priors) for i in ii])
 methods={'Model':ranks[ii],'Fixed June products':np.tile(pop_order,(len(ii),1)),'Keyword rule':keyword}
 def hit(rank,k):return (rank[:,:k]==y[ii,None]).any(axis=1)
 for name,rank in methods.items():
  for k in [1,2]:
   h=hit(rank,k);yes=int(h.sum());n=len(ii);lo,hi=ci(yes,n)
   shortlists.append(dict(period=tag,on='held under length-guarded rule',method=name,k=k,correct_in_list=yes,n=n,rate=yes/n,ci_low=lo,ci_high=hi))
  for j,l in enumerate(LABELS):
   h=hit(rank,2);t=y[ii]==j;n=int(t.sum());yes=int(h[t].sum());byq.append(dict(period=tag,method=name,true_product=l,correct_in_top2=yes,n=n,rate=yes/n if n else None))
 model_hits=hit(ranks[ii],2)
 for name in ['Fixed June products','Keyword rule']:
  other=hit(methods[name],2);diff=model_hits.astype(int)-other.astype(int);lo,hi=bootstrap(d.date.iloc[ii].to_numpy(),diff)
  paired.append(dict(period=tag,versus=name,held=len(ii),model_only=int((diff==1).sum()),baseline_only=int((diff==-1).sum()),paired_lift=float(diff.mean()),day_cluster_lower=lo,day_cluster_upper=hi))
def main(fetch, refit=False):
 OUT.mkdir(exist_ok=True)
 for name,sha in [('protocol.json','9770bce3c8aeed2e778771281b76bd91502856d33c0dd5d8341611e96264c2c5'),('backtest_protocol.json','03158956b5ff14b80b78148b2edff0bd6af58d9880198374909a9941b1c453ad')]:
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
 policy_sha=hashlib.sha256((ROOT/'experiments/policy/locked_policy.json').read_bytes()).hexdigest()
 assert policy_sha=='2dc5b756de0255a9b2fd863d9981be2902b0064bc81868076a9a57b6822f0aa8'
 raws={};man={}
 for m in FILES:raws[m],man[m]=read_month(m,fetch)
 hist=pd.read_pickle(DATA/'cohort.pkl')
 prior=pd.concat([hist[['id','text']],raws['june'][['id','text']],raws['july'][['id','text']]],ignore_index=True).drop_duplicates('id')
 prior_count=len(prior);may=raws['may']
 may=may.assign(key=may.id.astype(str).map(lambda s:hashlib.blake2b(('relay-may-frozen:'+s).encode(),digest_size=8).hexdigest())).sort_values('key').head(6000).sort_values(['date','id']).reset_index(drop=True)
 rawprior=raws['june'];prior_counts=rawprior.loc[rawprior.date<'2026-06-08','product'].map(label).value_counts().reindex(LABELS,fill_value=0).astype(int).tolist()
 periods,summ=dedup({'may':may,'august':raws['august']},prior);del prior,raws,rawprior;gc.collect()
 for m in summ:man[m].update(summ[m])
 man['may']['sampled']=len(may);man['august']['sampled']=man['august']['eligible']
 man['reference_ids']=prior_count;man['fixed_june_rank_counts']=dict(zip(LABELS,prior_counts));man['policy_hash']=policy_sha
 man['archive_hashes']={m:h for m,(_,h) in FILES.items()}
 dump(man,OUT/'source_audit.json')
 jp=DATA/'blind_july';d=pd.read_pickle(jp/'blind_inputs.pkl').rename(columns={'word_count':'words'})
 with np.load(jp/'july_labels.npz') as z:
  assert np.array_equal(d.id.to_numpy(),z['id']);yj=z['y'].astype(int)
 assert len(d)==8830
 transfer=[];ranks=[];byq=[];paired=[]
 for m in ['may','august']:
  g=periods[m];summarize(m.title(),g,g['product'].map(label).map(dict(zip(LABELS,range(7)))).to_numpy(dtype=int),prior_counts,transfer,ranks,byq,paired)
 summarize('July',d,yj,prior_counts,transfer,ranks,byq,paired)
 for name,rows in [('gates',transfer),('reviewer_shortlist',ranks),('shortlist_by_product',byq),('paired_shortlist',paired)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
 july_original=next(r for r in transfer if (r['period'],r['policy'],r['group'])==('July','June rule','All'))
 observed = [int(july_original[k]) for k in ('group_n','routed','wrong')]
 reference = [8830, 928, 17]
 if refit:
  dump({'mode':'local refit; frozen public June gate applied to refitted base scores',
        'fields':['eligible','flagged','label_disagreements'],'published':reference,'observed':observed,
        'exact_count_match':observed==reference},OUT/'refit_comparison.json')
  print('Local-refit counts:',observed,'; published counts:',reference)
 elif observed != reference:
  raise ValueError(f'Frozen replay mismatch: expected {reference}, got {observed}. For a separately fitted base model use --refit to report the difference.')
 print('Saved aggregate data only. August meaningful performance audit:',man['august']['retained']>=100)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fetch',action='store_true');ap.add_argument('--refit',action='store_true',help='Record score drift explicitly for a fresh model fit; input archives and protocols must still match.');args=ap.parse_args();main(args.fetch,args.refit)
