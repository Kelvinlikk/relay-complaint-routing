"""Local, read-only policy demonstration. Inference is in memory; no logs or transfers."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import argparse,json,os,sys,threading
import numpy as np,joblib
from scipy.special import softmax
from catboost import CatBoostClassifier
HERE=Path(__file__).resolve().parent
def choose_root(path):
 paths=[Path(path)] if path else [HERE.parent]
 paths+=list(HERE.parents)
 return next((p for p in paths if (p/'models'/'linear.joblib').is_file() and (p/'results'/'policy.json').is_file()),None)
def load_settings(p):
 return json.loads(p.read_text(encoding='utf-8'))
class Desk:
 def __init__(self,root,config=None):
  self.m=joblib.load(root/'models'/'linear.joblib');self.tab=CatBoostClassifier();self.tab.load_model(str(root/'models'/'tab.cbm'))
  self.old=load_settings(root/'results'/'policy.json');self.labels=self.old['labels']
  self.frozen=load_settings(config) if config and config.exists() else None
  self.samples=load_settings(HERE/'examples.json');self.lock=threading.Lock()
 def route(self,case):
  msg=case.get('text','').strip(); company=case.get('company','').strip();state=case.get('state','').strip().upper()
  if len(msg)>6000 or len(company)>160 or len(state)>5:raise ValueError('Input exceeds the local demo limits.')
  base={'decision':'Human review','state':'review','reason':'','queue':None,'confidence':None,'top':[],'evidence':'','legacy':None,'cut':None,'cert':None}
  if len(msg.split())<10:base['reason']='The text contains fewer than ten words. It falls outside the evaluated intake population.';return base
  t=self.old['temperatures'];cal=lambda v,temp:softmax(np.log(np.clip(v,1e-9,1))/temp)
  with self.lock:
   tx=cal(self.m.predict_proba([msg])[0],t['TF-IDF'])
   tab=cal(self.tab.predict_proba([[company or '[unknown]',state or '[unknown]']],thread_count=1)[0],t['CatBoost'])
  w=self.old['weights'];assert w['TinyBERT']==0
  p=cal(w['TF-IDF']*tx+w['CatBoost']*tab,t['Combined']) if company else tx
  i=int(p.argmax()); label=self.labels[i];cut=self.old['thresholds']['Combined' if company else 'TF-IDF']
  base['legacy']=label if i!=6 and p[i]>=cut else 'Human review'
  base['raw_model']=('text + known mailbox' if company else 'text only')
  base.update({'queue':label,'confidence':float(p[i]),'top':[{'name':self.labels[j],'v':float(p[j])} for j in np.argsort(-p)[:3]]})
  if not company:
   base['reason']='No independently verified client mailbox. The text-only suggestion is available to a reviewer; automatic routing is disabled.';return base
  if not self.frozen:
   base['reason']='The guarded policy has not been loaded. Keep this case with a reviewer.';return base
  return apply_guard(base,p,self.frozen,self.labels)
def apply_guard(base,p,config,labels):
 """Final guarded scoring overlay, deliberately fail-closed for absent evidence."""
 assert config['labels']==labels
 v=softmax(np.log(np.clip(p,1e-9,1))/float(config['temperature'])+np.array(config['class_offsets']))
 j=int(v.argmax());label=labels[j]
 base.update({'queue':label,'confidence':float(v[j]),'top':[{'name':labels[i],'v':float(v[i])} for i in np.argsort(-v)[:3]]})
 if j==6:
  base['reason']='This complaint is outside the six specialist queues. A reviewer must determine the destination.'
  base['evidence']='Other is always reserved for review.';return base
 cert=config['certificates'][j];cut=float(config['queue_cutoffs'][j]);base.update(cut=cut,cert=cert)
 base['evidence']=('June label audit: '+str(cert['wrong'])+' label mismatches / '+str(cert['n'])+' proposed handoffs.' if cert['n'] else 'June label audit: no eligible candidate routes for this destination.')
 if not cert['enabled']:
  base['reason']=f'Suggested queue: {label}. This destination did not pass its independent 5% error test. A person must confirm the route.';return base
 if v[j]<cut:
  base['reason']=f'Suggested queue: {label}. The case does not clear the {cut:.0%} score cutoff for this destination.';return base
 base.update({'decision':label,'state':'route','reason':f'The {label} audit passed and this case clears its {cut:.0%} cutoff. This is a simulated policy decision.'})
 return base
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo');ap.add_argument('--port',type=int,default=8765);ap.add_argument('--policy',default=str(HERE/'guard.json'));opt=ap.parse_args()
 root=choose_root(opt.repo)
 if not root:sys.exit('Cannot find the model bundle. Run python demo/desk.py --repo /path/to/Complaint_Routing')
 desk=Desk(root,Path(opt.policy)); evidence_path=HERE/'evidence.json'
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,data,mime='application/json',status=200):
   data=data if isinstance(data,bytes) else json.dumps(data).encode()
   self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
  def do_GET(self):
   p=urlparse(self.path).path
   if p=='/':return self.send((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
   if p=='/api/setup':return self.send({'samples':desk.samples,'evidence':load_settings(evidence_path) if evidence_path.exists() else {},'labels':desk.labels})
   return self.send({'error':'Not found'},status=404)
  def do_POST(self):
   if self.path!='/api/route':return self.send({'error':'Not found'},status=404)
   try:
    size=int(self.headers.get('Content-Length','0'))
    if size>36000 or size<=0:raise ValueError('Invalid size.')
    d=json.loads(self.rfile.read(size))
    if not isinstance(d,dict) or any(not isinstance(d.get(k,''),str) for k in ['text','company','state']):raise ValueError('Expected text fields.')
    return self.send(desk.route(d))
   except (TypeError,ValueError,UnicodeError) as ex:return self.send({'error':str(ex)},status=400)
 print('Review desk: http://127.0.0.1:%d (local only; no input is logged)'%opt.port,flush=True)
 ThreadingHTTPServer(('127.0.0.1',opt.port),Handler).serve_forever()
if __name__=='__main__':main()
