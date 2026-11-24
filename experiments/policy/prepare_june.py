"""Build expanded June from the official archive; no July access. Histories preserved."""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
import numpy as np,pandas as pd,joblib
from scipy.special import softmax
from catboost import CatBoostClassifier
from datasketch import MinHash,MinHashLSH

def prepare(repo,out):
    sys.path.insert(0,str(repo/'src')); from prepare import clean,shingles,label,FILES,LABELS
    priv=out/'private';priv.mkdir(parents=True,exist_ok=True)
    d=pd.read_pickle(repo/'data/cohort.pkl')
    h=d.query("split != 'shift'")[['id','text']].copy()
    fn=repo/'data'/FILES['shift']
    with zipfile.ZipFile(fn) as a:
        df=pd.read_csv(a.open(a.namelist()[0]),usecols=['Date received','Product','Consumer complaint narrative','Company','State','Complaint ID'],low_memory=False)
    df.columns=[{'Date received':'date','Consumer complaint narrative':'text','Company':'company','State':'state','Complaint ID':'id','Product':'product'}[v] for v in df]
    df=df.dropna(subset=['text']).copy();df=df[df.text.str.split().str.len().ge(10)].sort_values(['date','id']).reset_index(drop=True)
    seen=set();sets={};lsh=MinHashLSH(threshold=.73,num_perm=64);m=MinHash(num_perm=64,seed=15071);a,b=m.permutations
    def retain(s,idx):
        ss=clean(s);sig=hashlib.sha256(ss.encode()).digest()
        if sig in seen:return False
        v=shingles(ss)
        if not v:return False
        arr=np.array(sorted(v),dtype=np.uint64)
        m.hashvalues=(((arr[:,None]*a+b)%np.uint64((1<<61)-1))&np.uint64((1<<32)-1)).min(axis=0)
        if any(len(v & sets[j])>=.80*len(v|sets[j]) for j in lsh.query(m)):return False
        seen.add(sig);lsh.insert(idx,m);sets[idx]=v;return True
    for v in h.itertuples(index=False):retain(v.text,int(v.id))
    retained=[retain(v.text,int(v.id)) for v in df.itertuples(index=False)]
    arr=df.loc[retained].copy();tags=arr['product'].map(label)
    arr['y']=tags.map({k:i for i,k in enumerate(LABELS)})
    assert arr.y.notna().all() and arr.date.str.startswith('2026-06').all()
    del sets,lsh,seen,df,h,d
    txt=joblib.load(repo/'models/linear.joblib').predict_proba(arr.text)
    tab=CatBoostClassifier();tab.load_model(str(repo/'models/tab.cbm'))
    pp=tab.predict_proba(arr[['company','state']].fillna('[unknown]').astype(str),thread_count=2)
    pol=json.loads((repo/'results/policy.json').read_text());tt=pol['temperatures'];w=pol['weights'];assert w['TinyBERT']==0
    norm=lambda q,t:softmax(np.log(np.clip(q,1e-9,1))/t,axis=1)
    legacy=norm(w['TF-IDF']*norm(txt,tt['TF-IDF'])+w['CatBoost']*norm(pp,tt['CatBoost']),tt['Combined'])
    np.savez_compressed(priv/'june_scores.npz',id=arr.id.to_numpy(),date=arr.date.to_numpy(dtype='U10'),y=arr.y.to_numpy(dtype=np.int64),p_legacy=legacy)
    (out/'june_cohort.json').write_text(json.dumps(dict(raw_eligible=len(retained),retained=len(arr),excluded_duplicates=len(retained)-len(arr),archive_sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),retained_id_hash=hashlib.sha256(','.join(arr.id.astype(str)).encode()).hexdigest(),public_raw_text_redistributed=False),indent=2))
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--repo',type=Path,required=True);pa.add_argument('--out',type=Path,default=Path(__file__).parent);q=pa.parse_args();prepare(q.repo.resolve(),q.out.resolve())
