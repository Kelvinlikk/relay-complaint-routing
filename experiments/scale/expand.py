"""From project root after restoring checkpoint edition + base datasets. See README."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import csv, gc, hashlib, json, sys, zipfile, time
from collections import defaultdict, Counter
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.metrics import f1_score
from datasketch import MinHash
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments/transfer')]
import replay as rp
from prepare import LABELS,clean,shingles,get_file
from july_holdout import mkhash
D=ROOT/'data'; PRIVATE=D/'scale';PRIVATE.mkdir(exist_ok=True)
def log(*x):print(time.strftime('%H:%M:%S'),*x,flush=True)
def key(x,seed):return hashlib.blake2b((seed+str(x)).encode(),digest_size=8).hexdigest()
def fingerprint(d):return hashlib.sha256(','.join(d.id.astype(str)).encode()).hexdigest()
def write(o,p):p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
perm=MinHash(num_perm=64,seed=15071).permutations
class Index:
 def __init__(self):self.ids=set();self.exacts=set();self.bands=[defaultdict(list) for _ in range(8)];self.words=[]
 def put(self,id,text,screen):
  if id in self.ids:return 'id'
  s=clean(text);h=hashlib.sha256(s.encode()).digest()
  if h in self.exacts:
   if not screen:self.ids.add(id)
   return 'exact'
  v=shingles(s);arr=np.fromiter(v,dtype=np.uint32);arr.sort()
  z=mkhash(v,perm).hashvalues.astype(np.uint32).reshape(8,8)
  bands=[x.tobytes() for x in z]
  if screen:
   hits=set()
   for j,b in enumerate(bands):hits.update(self.bands[j].get(b,()))
   for i in hits:
    w=self.words[i]
    if min(len(w),len(arr))<.8*max(len(w),len(arr)):continue
    n=np.intersect1d(w,arr,assume_unique=True).size
    if n>=.8*(len(w)+len(arr)-n):return 'near'
  i=len(self.words);self.words.append(arr);self.exacts.add(h);self.ids.add(id)
  for j,b in enumerate(bands):self.bands[j][b].append(i)
  return 'keep'
 def run(self,d,screen=True):
  ans=[];loss=Counter()
  for r in d.itertuples(index=False):
   decision=self.put(r.id,r.text,screen);ans.append(decision=='keep');loss[decision]+=1
  return d.loc[np.asarray(ans)].reset_index(drop=True),dict(loss)
def archive():
 chunks=[]
 with zipfile.ZipFile(get_file('CCDB_Export_2_May_2018_through_April_2021.zip')) as z:
  for g in pd.read_csv(z.open(z.namelist()[0]),usecols=list(rp.COLS),chunksize=40000,low_memory=False):
   g=g.rename(columns=rp.COLS);g=g[g.text.notna() & ((g.date<'2020-01')|(g.date>='2020-09'))].copy();g=g[g.text.str.split().str.len()>=10];chunks.append(g)
 g=pd.concat(chunks,ignore_index=True);del chunks
 g['month']=g.date.str[:7];g['label']=g['product'].map(rp.label)
 return g
def prep():
 assert hashlib.sha256((HERE/'protocol.md').read_bytes()).hexdigest()=='e46850c087c9b26db1c653b571ba0282c92d9f3f4a62c38c86e72d53b0cbed7b'
 assert hashlib.sha256((ROOT/'experiments/policy/locked_policy.json').read_bytes()).hexdigest()=='2dc5b756de0255a9b2fd863d9981be2902b0064bc81868076a9a57b6822f0aa8'
 base=pd.read_pickle(D/'cohort.pkl');ref=Index();ref.run(base,False)
 raws={}
 for m in ['may','june','july','august']:raws[m],_=rp.read_month(m,False)
 for m in ['june','july']:ref.run(raws[m],False)
 may=raws['may'].copy();may['ord']=may.id.map(lambda i:key(i,'relay-may-frozen:'));first=may.sort_values('ord').head(6000).sort_values(['date','id']);rest=may.loc[~may.id.isin(first.id)].sort_values(['date','id'])
 a,af=ref.run(first);prev=json.loads((ROOT/'experiments/transfer/results/source_audit.json').read_text());assert len(a)==5254 and fingerprint(a)==prev['may']['retained_id_sha256']
 b,bf=ref.run(rest);log('May original/additional',len(a),len(b))
 for part,g in [('sample',a),('additional',b)]:g.assign(member=part).to_pickle(PRIVATE/(part+'.pkl'))
 ref.run(may,False);ref.run(raws['august'],False);del raws,may;gc.collect()
 g=archive();train=g[g.date<'2020-01'].copy();train['ord']=train.id.map(lambda i:key(i,'15071:'));train=train.sort_values('ord').iloc[26000:62000].sort_values(['date','id']);assert len(train)==36000
 add,tf=ref.run(train);add.to_pickle(PRIVATE/'train.pkl');log('New training cases;',len(add))
 test=g[g.date>='2020-09'].copy();test['ord']=test.id.map(lambda i:key(i,'relay-scale-test-v1:'));test=test.sort_values('ord').groupby('month',sort=True).head(2000).sort_values(['date','id']);assert len(test)==16000
 newtest,ef=ref.run(test);newtest.to_pickle(PRIVATE/'test.pkl');log('New historical test cases;',len(newtest))
 assert not(set(add.id)&set(newtest.id)) and not(set(base.id)&(set(add.id)|set(newtest.id)))
 write(dict(may_old=af,may_additional=bf,train_added=tf,test_new=ef,hashes={k:fingerprint(v) for k,v in [('added_train',add),('fresh_test',newtest),('original_may',a),('extra_may',b)]}),OUT/'selection.json')
def may_eval():
 a=pd.read_pickle(PRIVATE/'sample.pkl');b=pd.read_pickle(PRIVATE/'additional.pkl');audit=json.loads((ROOT/'experiments/transfer/results/source_audit.json').read_text());priors=[audit['fixed_june_rank_counts'][k] for k in LABELS]
 for nm,l in [('original',a),('additional',b),('all',pd.concat([a,b],ignore_index=True))]:
  y=l['product'].map(rp.label).map(LABELS.index).to_numpy()
  gate=[];ranks=[];byq=[];paired=[];rp.summarize('May '+nm,l,y,priors,gate,ranks,byq,paired)
  for name,rows in [('gates',gate),('rank',ranks),('per_product',byq),('paired_shortlist',paired)]:pd.DataFrame(rows).to_csv(OUT/(nm+'_'+name+'.csv'),index=False)
  log('May',nm,gate[0]['routed'],gate[0]['wrong'])
def paired(y,old,new,dates):
 _,day=np.unique(dates,return_inverse=True); n=day.max()+1;ma=[]
 for pred in (old,new):
  ma.append(np.bincount(day*49+y*7+pred,minlength=n*49).reshape(n,7,7))
 def f(x):
  den=x.sum(axis=1)+x.sum(axis=2);return np.divide(2*x.diagonal(axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(axis=1)
 rng=np.random.default_rng(15071);weights=rng.multinomial(n,np.ones(n)/n,size=2000);delts=f((weights@ma[1].reshape(n,49)).reshape(-1,7,7))-f((weights@ma[0].reshape(n,49)).reshape(-1,7,7))
 return np.quantile(delts,[.025,.975]).tolist()
def fit():
 from threadpoolctl import threadpool_limits
 base=pd.read_pickle(D/'cohort.pkl');train0=base[base.split=='train'];add=pd.read_pickle(PRIVATE/'train.pkl');train=pd.concat([train0,add],ignore_index=True);orig=joblib.load(ROOT/'models/linear.joblib');big=clone(orig)
 log('Fitting',len(train),'vs',len(train0))
 with threadpool_limits(limits=2):
  big.fit(train.text,train.label.map(LABELS.index));log('fit complete',big.steps[-1][1].n_iter_.tolist())
  joblib.dump(big,PRIVATE/'challenger.joblib')
  jj=D/'blind_july';jul=pd.read_pickle(jj/'blind_inputs.pkl')
  with np.load(jj/'july_labels.npz') as yy:assert np.array_equal(jul.id,yy['id']);jul['label']=[LABELS[i] for i in yy['y']]
  newtest=pd.read_pickle(PRIVATE/'test.pkl');out=[];months=[]
  for k,d in [('original historical test',base[base.split=='test']),('additional historical test',newtest),('previously opened July',jul)]:
   y=d.label.map(LABELS.index).to_numpy();p0=orig.predict(d.text);p1=big.predict(d.text);lo,hi=paired(y,p0,p1,d.date.to_numpy())
   def metrics(y,p):return f1_score(y,p,average='macro',labels=list(range(7)),zero_division=0)
   s0=metrics(y,p0);s1=metrics(y,p1);out.append(dict(test=k,n=len(d),old_f1=s0,new_f1=s1,delta=s1-s0,lo=lo,hi=hi,old_accuracy=float(np.mean(y==p0)),new_accuracy=float(np.mean(y==p1))))
   for month in sorted(set(d.date.str[:7])):
    mask=(d.date.str[:7]==month).to_numpy();months.append(dict(test=k,month=month,n=int(mask.sum()),old_f1=metrics(y[mask],p0[mask]),new_f1=metrics(y[mask],p1[mask])))
  pd.DataFrame(out).to_csv(OUT/'learning.csv',index=False);pd.DataFrame(months).to_csv(OUT/'months.csv',index=False)
  write(dict(original_training=len(train0),additional_training=len(add),expanded_training=len(train),C=big.steps[-1][1].C,solver=big.steps[-1][1].solver,iterations=big.steps[-1][1].n_iter_.tolist(),iteration_cap=big.steps[-1][1].max_iter),OUT/'fit.json')
if __name__=='__main__':
 for step in (sys.argv[1:] or ['prep','may','fit']):{'prep':prep,'may':may_eval,'fit':fit}[step]()
