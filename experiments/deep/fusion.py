"""Train, ablate or score a joint complaint narrative + intake-metadata model.
Run from any directory. Input archives stay local. Only aggregates/model files are public.
"""
from pathlib import Path
import argparse,os,json,math,re
os.environ.setdefault("USE_TF","0");os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import joblib,numpy as np,pandas as pd
from scipy import sparse
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler,OneHotEncoder,normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,f1_score,accuracy_score
import warnings
LABELS=['Accounts','Cards','Collections','Credit reports','Mortgages','Transfers','Other / review']
HEADS={'Intake':['meta'],'Encoder':['emb'],'Encoder + intake':['emb','meta'],
       'Sparse + intake':['tf','meta'],'Joint':['tf','emb','meta']}
def encode(frame,models,threads=2):
 import torch
 from transformers import BertTokenizerFast,BertForSequenceClassification
 torch.set_num_threads(threads)
 tok=BertTokenizerFast.from_pretrained(str(models/'fine'));model=BertForSequenceClassification.from_pretrained(str(models/'fine')).bert.eval()
 acc=[]
 with torch.inference_mode():
  for start in range(0,len(frame),64):
   texts=frame.text.iloc[start:start+64].tolist()
   enc=tok(texts,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
   ids=np.zeros((len(enc),112),np.int64)
   for i,t in enumerate(enc):
    z=t if len(t)<=110 else t[:78]+t[-32:];seq=[101]+z+[102];ids[i,:len(seq)]=seq
   x=torch.from_numpy(ids)
   # Pre-classifier, 128-dimensional pooled [CLS] vector after the BERT pooler's tanh.
   acc.append(model(input_ids=x,attention_mask=x.ne(0)).pooler_output.numpy().astype(np.float32))
 return np.concatenate(acc)
def cats(f):
 return f[['company','state']].fillna('UNKNOWN').astype(str)
def shape(f):
 rows=[]
 for text in f.text:
  words=text.split();n=max(len(words),1);nc=max(len(text),1)
  rows.append([np.log1p(n),np.log1p(nc),len(set(w.lower() for w in words))/n,
               sum(ch.isdigit() for ch in text)/nc,sum(ch.isupper() for ch in text)/nc,
               np.log1p(text.count('?')),np.log1p(text.count('$')), np.log1p(text.count('\n'))])
 return np.asarray(rows,dtype=np.float32)
def mk_maps(frame,emb,train):
 return dict(emb_scaler=StandardScaler().fit(emb[train]),
  cat=OneHotEncoder(handle_unknown='ignore',min_frequency=3,dtype=np.float32).fit(cats(frame.iloc[train])),
  shape_scaler=StandardScaler().fit(shape(frame.iloc[train])))
def parts(frame,emb,base,mp):
 vec=joblib.load(base/'models'/'linear.joblib').steps[0][1]
 m=sparse.hstack([mp['cat'].transform(cats(frame)),sparse.csr_matrix(np.clip(mp['shape_scaler'].transform(shape(frame)),-4,4)*.10)],format='csr',dtype=np.float32)
 return dict(tf=vec.transform(frame.text).astype(np.float32),
             emb=sparse.csr_matrix(normalize(mp['emb_scaler'].transform(emb)),dtype=np.float32),
             meta=m)
def mat(blocks,names):
 return sparse.hstack([blocks[x] for x in names],format='csr',dtype=np.float32)
def bnd(e,n):
 if n==0:return 1.
 p=e/n; z=1.959963984540054
 return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
def accept(prob,t):
 return (np.max(prob,axis=1)>=t)&(np.argmax(prob,axis=1)!=6)
def cutoff(prob,labels):
 cands=[]
 for t in np.arange(.5,.996,.005):
  a=accept(prob,t); n=a.sum();e=(prob.argmax(1)[a]!=labels[a]).sum()
  if n>=100 and bnd(e,n)<=.05:cands.append((n,-float(t)))
 return -max(cands)[1] if cands else 1.01
def unchanged(base):
 v=json.loads((base/'results'/'policy.json').read_text());c=np.load(base/'data'/'classic.npz')
 p1=softmax(np.log(np.clip(c['text'],1e-9,1))/v['temperatures']['TF-IDF'],axis=1)
 p2=softmax(np.log(np.clip(c['tab'],1e-9,1))/v['temperatures']['CatBoost'],axis=1)
 w=v['weights'];assert w['TinyBERT']==0
 return softmax(np.log(np.clip(w['TF-IDF']*p1+w['CatBoost']*p2,1e-9,1))/v['temperatures']['Combined'],axis=1),v['thresholds']['Combined']
def histstats(pred,y):
 return dict(macro_f1=float(f1_score(y,pred,labels=range(7),average='macro')),accuracy=float(np.mean(pred==y)))
def fit(args):
 base=args.repo;w=args.out;w.mkdir(exist_ok=True,parents=True); frame=pd.read_pickle(base/'data'/'cohort.pkl').reset_index(drop=True)
 em=w/'private_embeddings.npz'
 if em.exists():
  z=np.load(em);assert np.array_equal(z['id'],frame.id);emb=z['emb']
 else:
  emb=encode(frame,base/'models',args.threads);np.savez_compressed(em,id=frame.id.to_numpy(),emb=emb)
 splits={k:np.flatnonzero(frame.split.values==k) for k in ['train','tune','policy','test','shift']};tr=splits['train'];tu=splits['tune'];po=splits['policy']; y=frame.label.map({k:i for i,k in enumerate(LABELS)}).to_numpy()
 mp=mk_maps(frame,emb,tr);data=parts(frame,emb,base,mp);preds={};details={}
 # Predeclared C grid; frozen base encoders, no tune/test rows in head/feature-statistic fitting.
 opts={}
 for k,names in HEADS.items():
  print('head',k,flush=True);X=mat(data,names);best=None
  for C in [1.,8.]:
   lr=LogisticRegression(C=C,solver='lbfgs',max_iter=250,tol=1e-4)
   lr.fit(X[tr],y[tr]);pro=lr.predict_proba(X[tu]);score=log_loss(y[tu],pro,labels=range(7));print(k,C,score,flush=True)
   if best is None or score<best[0]:best=(score,C,lr)
  loss,C,lr=best; logits=lr.decision_function(X)
  T=float(minimize_scalar(lambda t:log_loss(y[tu],softmax(logits[tu]/t,axis=1),labels=range(7)),bounds=(.3,4),method='bounded').x)
  prob=softmax(logits/T,axis=1);cut=cutoff(prob[po],y[po]);opts[k]=(float(C),T,cut)
  details[k]={'C':float(C),'temperature':T,'policy_cutoff':float(cut),'validation_nll':float(loss),'inputs':names}
  preds[k]=prob;print('done',k,flush=True)
  if k in ('Joint','Encoder + intake','Sparse + intake'):
   joblib.dump(dict(lr=lr,maps=mp,temp=T,cut=cut,names=names,labels=LABELS),w/({'Joint':'joint','Encoder + intake':'deep_intake','Sparse + intake':'sparse_intake'}[k]+'.joblib'),compress=3)
 original,oldcut=unchanged(base);preds['Original blend']=original;opts['Original blend']=(None,None,oldcut);metrics=[];queue=[];splitids={}
 for k,p in preds.items():
  for split in ['tune','policy','test','shift']:
   i=splits[split];t=opts[k][2];v=p[i];yp=v.argmax(1);a=accept(v,t);err=int(np.sum(a&(yp!=y[i])));n=int(a.sum())
   metrics.append(dict(model=k,period=split,n=len(i),**histstats(yp,y[i]),nll=log_loss(y[i],v,labels=range(7)),threshold=t,auto=n,wrong=err,coverage=n/len(i),misroute=err/n if n else None))
   for j,label in enumerate(LABELS):
    g=y[i]==j;queue.append(dict(model=k,period=split,label=label,n=int(g.sum()),recall=float(np.mean(yp[g]==y[i][g]))))
 for split in ['test','shift']:
  ix=splits[split];correct=y[ix];rnd=np.random.default_rng(15071);cache={k:v[ix].argmax(1) for k,v in preds.items()}
  for one,two in [('Joint','Sparse + intake'),('Joint','Original blend'),('Encoder + intake','Encoder')]:
   d=[]
   for j in range(500):
    s=rnd.integers(0,len(ix),size=len(ix));y0=correct[s]
    d.append(f1_score(y0,cache[one][s],labels=range(7),average='macro')-f1_score(y0,cache[two][s],labels=range(7),average='macro'))
   k=f'{split}: {one} minus {two}';splitids[k]={'delta_macro_f1':float(f1_score(correct,cache[one],labels=range(7),average='macro')-f1_score(correct,cache[two],labels=range(7),average='macro')),'percentile95_iid_paired':np.quantile(d,[.025,.975]).tolist()}
 pd.DataFrame(metrics).to_csv(w/'benchmarks.csv',index=False);pd.DataFrame(queue).to_csv(w/'recall_by_queue.csv',index=False)
 (w/'head_config.json').write_text(json.dumps(details,indent=2));(w/'paired_differences.json').write_text(json.dumps(splitids,indent=2));print(json.dumps(splitids),flush=True)
def score_frame(frame,repo,weights,threads=2,emb=None):
 # Deliberately takes only intake-available inputs; no complaint IDs, dates, labels or outcomes in the model.
 frame=frame.loc[:,['text','company','state']].reset_index(drop=True)
 file=joblib.load(weights)
 if emb is None:emb=encode(frame,Path(repo)/'models',threads)
 data=parts(frame,emb,Path(repo),file['maps']);return softmax(file['lr'].decision_function(mat(data,file['names']))/file['temp'],axis=1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
 fit(p.parse_args())
