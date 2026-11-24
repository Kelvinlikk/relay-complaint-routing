"""Blind comparison: score first, release aggregate comparisons only after parent freeze.
Requires an exact matching, separately provided held-out truth file for the release step.
"""
import argparse, json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import softmax
from sklearn.metrics import f1_score,log_loss
from fusion import encode,parts,mat,LABELS,accept,bnd
def score(args):
 frame=pd.read_pickle(args.inputs)
 if 'label' in frame:raise ValueError('Blind scorer must not receive validation labels.')
 src=frame.loc[:,['text','company','state']]
 if args.embedding_cache and args.embedding_cache.exists():
  z=np.load(args.embedding_cache);assert np.array_equal(z['id'],frame.id);e=z['emb']
 else:e=encode(src,args.repo/'models',args.threads)
 out={'id':frame.id.to_numpy()}
 for name in ['joint','sparse_intake','deep_intake']:
  m=joblib.load(args.model_dir/(name+'.joblib'))
  v=softmax(m['lr'].decision_function(mat(parts(src,e,args.repo,m['maps']),m['names']))/m['temp'],axis=1)
  out[name]=v
 args.scores.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(args.scores,**out)
 print('Blind score rows:',len(frame),'file sha256:',hashlib.sha256(args.scores.read_bytes()).hexdigest())
def audit(args):
 if not args.labels:raise ValueError('Audit requires an explicit label file.')
 if args.labels.suffix=='.npz':
  data=np.load(args.labels);d=pd.DataFrame({'id':data['id'],'label':data['y'].astype(np.int64)})
 elif args.labels.suffix=='.pkl':d=pd.read_pickle(args.labels)
 else:d=pd.read_csv(args.labels)
 scores=np.load(args.scores,allow_pickle=False)
 y=pd.Series(d.label.values,index=d.id.astype(str)).loc[list(map(str,scores['id']))]
 if y.isna().any():raise ValueError('Missing labels')
 if str(y.dtype) not in ['int32','int64']:y=y.map(dict(zip(LABELS,range(7))))
 y=y.to_numpy(dtype=int);rows=[]
 for k in ('joint','sparse_intake','deep_intake'):
  vv=scores[k];cfg=joblib.load(args.model_dir/(k+'.joblib'));t=cfg['cut'];yp=vv.argmax(1);ac=accept(vv,t);n=int(ac.sum());er=int((yp[ac]!=y[ac]).sum())
  rows.append(dict(model=k,n=len(y),macro_f1=f1_score(y,yp,labels=range(7),average='macro'),accuracy=float(np.mean(y==yp)),nll=log_loss(y,vv,labels=range(7)),threshold=t,auto=n,wrong=er,coverage=n/len(y),misroute=er/n if n else None,wilson_95_upper=bnd(er,n)))
 pd.DataFrame(rows).to_csv(args.model_dir/'july_challengers.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['score','audit']);p.add_argument('--inputs',type=Path);p.add_argument('--labels',type=Path);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--embedding-cache',type=Path); a=p.parse_args();(score if a.command=='score' else audit)(a)
