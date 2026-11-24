"""Prepare a blind, next-month CFPB test. All private row-level outputs stay out of git.
From repository root after run.sh:
python experiments/external/july_holdout.py --repo . --output data/blind_july
Never use July predictions or labels to choose policy controls.
"""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, hashlib, json, sys, zipfile, gc
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.special import softmax
from datasketch import MinHash, MinHashLSH

MINHASH_SCHEME = MinHash(num_perm=64,seed=15071).scheme
URL = 'https://files.consumerfinance.gov/f/documents/CCDB_Export_20_July_2026.zip'
SOURCE = 'https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/'
def hashfile(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def get_source(out):
    path=out/URL.rsplit('/',1)[1]
    if not path.is_file():
        tmp=path.with_suffix('.part')
        with requests.get(URL,stream=True,timeout=150) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for b in r.iter_content(1<<20):f.write(b)
        tmp.replace(path)
    if hashfile(path)!='aafebbaaf0962c3d29532411ac2133c01bae830b7ea6ed1931a6e42278437d18':
        raise ValueError('Unexpected July snapshot bytes: do not silently replace the frozen evaluation source.')
    return path

def mkhash(v,perm):
    a,b=perm
    values=np.array(sorted(v),dtype=np.uint64)
    # Chunking also handles long published narratives without inflating peak memory.
    used=np.full(len(a),np.uint64((1<<32)-1))
    for i in range(0,len(values),512):
        vals=values[i:i+512,None]
        used=np.minimum(used,(((vals*a+b)%np.uint64((1<<61)-1)) & np.uint64((1<<32)-1)).min(axis=0))
    return MinHash(num_perm=64,seed=15071,hashvalues=used,permutations=perm,scheme=MINHASH_SCHEME)
def prep(repo,out):
    sys.path.insert(0,str(repo/'src'));from prepare import LABELS,label,clean,shingles
    src=get_source(out)
    cols=['Date received','Product','Consumer complaint narrative','Company','State','Complaint ID']
    renamed=dict(zip(cols,['date','product','text','company','state','id']))
    with zipfile.ZipFile(src) as z:
        member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        chunks=[]
        total=0
        for df in pd.read_csv(z.open(member),usecols=cols,chunksize=30000,low_memory=False):
            total+=len(df);df=df.rename(columns=renamed)
            assert df.date.str.startswith('2026-07-').all(), 'Unexpected source month'
            df=df.dropna(subset=['text']).copy()
            df['word_count']=df['text'].astype(str).str.split().str.len()
            df=df[df.word_count>=10];chunks.append(df)
    jul=pd.concat(chunks,ignore_index=True).sort_values(['date','id']).reset_index(drop=True)
    assert jul.id.is_unique
    prior_sample=pd.read_pickle(repo/'data'/'cohort.pkl')[['id','date','text']]
    # Include ALL June texts, independent of any adaptation-time dedup/selection.
    # This intentionally over-excludes overlapping templates rather than allowing
    # a near duplicate from ANY eligible June complaint to leak into July.
    june_name='CCDB_Export_19_June_2026.zip';june_chunks=[]
    with zipfile.ZipFile(repo/'data'/june_name) as z:
        june_member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        for part in pd.read_csv(z.open(june_member),usecols=['Date received','Consumer complaint narrative','Complaint ID'],chunksize=30000,low_memory=False):
            part=part.rename(columns=renamed).dropna(subset=['text'])
            part=part[part['text'].astype(str).str.split().str.len().ge(10)]
            june_chunks.append(part[['id','date','text']])
    june=pd.concat(june_chunks,ignore_index=True)
    assert june.date.str.startswith('2026-06-').all()
    prior=pd.concat([prior_sample,june],ignore_index=True).drop_duplicates('id')
    assert prior.id.is_unique
    assert prior.date.max()<jul.date.min()
    assert not set(prior.id).intersection(jul.id)
    lsh=MinHashLSH(threshold=.73,num_perm=64);perm=MinHash(num_perm=64,seed=15071).permutations
    signatures={};exact={};prior_ids=set(prior.id);reasons=Counter();kept=[]
    def index(i,s,v,m):
        exact[s]=i; lsh.insert(i,m);signatures[i]=v
    for r in prior.itertuples(index=False):
        s=clean(r.text);v=shingles(s);sig=hashlib.sha256(s.encode()).digest()
        index(r.id,sig,v,mkhash(v,perm))
    for r in jul.itertuples():
        s=clean(r.text);sig=hashlib.sha256(s.encode()).digest()
        if sig in exact:
            reasons['exact_prior' if exact[sig] in prior_ids else 'exact_july']+=1;continue
        v=shingles(s);m=mkhash(v,perm)
        match=[j for j in lsh.query(m) if len(v.intersection(signatures[j]))>=.80*len(v.union(signatures[j]))]
        if match:
            reasons['near_prior' if any(j in prior_ids for j in match) else 'near_july']+=1;continue
        kept.append(r.Index);index(r.id,sig,v,m)
    data=jul.iloc[kept].reset_index(drop=True)
    # No case-level labels in any feature or prediction file.
    y=np.array([LABELS.index(label(v)) for v in data['product']],dtype=np.int8)
    np.savez_compressed(out/'july_labels.npz',id=data.id.to_numpy('int64'),y=y)
    data[['id','date','text','company','state','word_count']].to_pickle(out/'blind_inputs.pkl')
    manifest={'source_url':URL,'source_page':SOURCE,'raw_zip_sha256':hashfile(src),'raw_archive_rows':total,
              'eligible_published_text_min10words':len(jul),'prior_unique_reference_rows':len(prior),
              'historical_and_original_june_reference_size':len(prior_sample),'all_june_eligible_reference_size':len(june),'prior_date_max':prior.date.max(),'date_min':data.date.min(),'date_max':data.date.max(),
              'sampling':'All published July narratives with at least 10 words, sorted by date and id. No filtering on product/company/outcome.',
              'deduplication':'All retained historical references plus all eligible June texts (even repeats), then earliest retained July, SHA-256 of canonical text and 64-permutation MinHash LSH on word trigrams, exact candidate Jaccard >= 0.8. Approximate candidates can miss similar items.',
              'removed':dict(reasons),'retained':len(data),'retained_id_sha256':hashlib.sha256(','.join(data.id.astype(str)).encode()).hexdigest()}
    assert sum(reasons.values())+len(data)==len(jul)
    (out/'july_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared blind records:',len(data),'(no label distribution inspected).',flush=True)
def predict(repo,out):
    import joblib,torch
    from transformers import BertTokenizerFast,BertForSequenceClassification
    from catboost import CatBoostClassifier
    d=pd.read_pickle(out/'blind_inputs.pkl')
    feat=d[['company','state']].fillna('[unknown]').astype(str)
    pipe=joblib.load(repo/'models'/'linear.joblib')
    ptext=np.concatenate([pipe.predict_proba(d['text'].iloc[k:k+1024]) for k in range(0,len(d),1024)])
    tab=CatBoostClassifier();tab.load_model(str(repo/'models'/'tab.cbm'));ptab=tab.predict_proba(feat,thread_count=2)
    masked=feat.copy();masked['company']='[unknown]';ptab_masked=tab.predict_proba(masked,thread_count=2)
    torch.set_num_threads(int(os.getenv('BERT_THREADS','2')))
    tok=BertTokenizerFast.from_pretrained(repo/'models'/'fine')
    model=BertForSequenceClassification.from_pretrained(repo/'models'/'fine');model.eval();logits=[]
    with torch.inference_mode():
        for i in range(0,len(d),128):
            enc=tok(d['text'].iloc[i:i+128].tolist(),truncation=False,add_special_tokens=False,verbose=False)
            batch=np.zeros((len(enc['input_ids']),112),np.int64)
            for j,t in enumerate(enc['input_ids']):
                t=t if len(t)<=110 else t[:78]+t[-32:]
                seq=[101]+t+[102];batch[j,:len(seq)]=seq
            x=torch.from_numpy(batch);logits.append(model(input_ids=x,attention_mask=x!=0).logits.numpy())
    logits=np.concatenate(logits); p=json.loads((repo/'results'/'policy.json').read_text())
    def temp(x,t):return softmax(np.log(np.clip(x,1e-9,1))/t,axis=1)
    pr={'TF-IDF':ptext,'CatBoost':ptab,'TinyBERT':softmax(logits,axis=1)}
    b=sum(p['weights'][k]*temp(pr[k],p['temperatures'][k]) for k in pr if p['weights'][k])
    old=temp(b,p['temperatures']['Combined'])
    data=dict(id=d.id.to_numpy('int64'),date=d.date.to_numpy(dtype=str),state=feat.state.to_numpy(dtype=str),
              company=feat.company.to_numpy(dtype=str),word_count=d.word_count.to_numpy(dtype=np.int32),
              p_text=ptext,p_tab=ptab,p_tab_masked=ptab_masked,logits_bert=logits,p_legacy=old)
    for arr in ('p_text','p_tab','p_tab_masked','p_legacy'):
        assert np.allclose(data[arr].sum(axis=1),1)
    np.savez_compressed(out/'july_predictions.npz',**data)
    print('Prediction file saved. No July outcomes read during scoring.',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['prepare','predict','all'],default='all')
    a=parser.parse_args();repo=a.repo.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.stage in ('prepare','all'):prep(repo,out)
    gc.collect()
    if a.stage in ('predict','all'):predict(repo,out)
