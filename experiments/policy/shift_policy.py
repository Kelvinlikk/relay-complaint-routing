"""Shift-calibrated routing. Frozen inputs are probabilities of a historical specialist model.
One public policy file; no train/test text, personal fields or complaint IDs in public outputs."""
from __future__ import annotations
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from scipy.stats import binom,beta,norm

LABELS=['Accounts','Cards','Collections','Credit reports','Mortgages','Transfers','Other / review']
RISK=.05; ALPHA=.05; GRID=np.arange(50,100,2)/100
VERSION='1.0';Z=norm.ppf(.975)
def transform(x,b,temperature):
    return softmax(np.log(np.clip(x,1e-9,1))/temperature+np.array(b)[None,:],axis=1)
def calibrate(x,y):
    """Seven centered class offsets and one global temperature; fixed L2 regularization .002."""
    lg=np.log(np.clip(x,1e-9,1)); prior_n=7
    def obj(theta):
        b=theta[:prior_n]-np.mean(theta[:prior_n]);logt=theta[-1];z=lg*np.exp(-logt)+b[None,:]
        nll=(logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean()
        return nll+.002*((b*b).sum()+logt*logt)
    opt=minimize(obj,np.zeros(8),method='L-BFGS-B',bounds=[(-5.,5.)]*7+[(-1.39,1.39)],options={'maxiter':300,'ftol':1e-12});assert opt.success,opt.message
    return list(opt.x[:7]-np.mean(opt.x[:7])),float(np.exp(opt.x[-1]))
def ci(k,n,z=Z):
    if n==0:return [None,None]
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;v=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,center-v)),float(min(1,center+v))]
def cp(k,n,alpha=.05):
    return None if n==0 else 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))
def candidates(x,y,per_queue=True):
    """Select broadest fixed-grid coverage satisfying 80% one-sided Wilson upper <=4%.
    This tuning guard is deliberately NOT the 5% launch certificate."""
    dest=x.argmax(1);score=x.max(1);cuts=[]
    for j in range(6) if per_queue else [None]:
        permitted=(dest<6) if j is None else (dest==j)
        rows=[]
        for th in GRID:
            accepted=permitted&(score>=th);n=int(accepted.sum());k=int((accepted&(dest!=y)).sum())
            if n>=30 and ci(k,n,norm.ppf(.8))[1]<=.04:rows.append((n,-float(th)))
        cuts.append(-max(rows)[1] if rows else 1.01)
    return cuts
def certify(x,y,cuts,per_queue,alpha=ALPHA):
    """Independent window; exact binomial H0 p >= 5% and Holm at family-wise 5%."""
    dest=x.argmax(1);scores=x.max(1);rows=[]
    for j,th in enumerate(cuts):
        filt=(dest==j if per_queue else dest<6)&(scores>=th)
        n=int(filt.sum());k=int((filt&(dest!=y)).sum())
        raw=float(binom.cdf(k,n,.05)) if n else 1.
        rows.append({'queue':LABELS[j] if per_queue else 'All specialist routes','candidate_cutoff':th,'n':n,'wrong':k,'p_exact':raw,'one_sided_95_upper':cp(k,n)})
    order=np.argsort([r['p_exact'] for r in rows],kind='stable');last=0.
    for rank,j in enumerate(order):
        last=max(last,min(1.,(len(rows)-rank)*rows[j]['p_exact']));rows[j]['holm_p']=float(last)
        rows[j]['enabled']=bool(last<=alpha);rows[j]['cutoff']=cuts[j] if last<=alpha else 1.01
    return rows
def apply(x,spec):
    """Pure serving interface; caller must verify tenant, language, urgency and length first."""
    x=np.atleast_2d(x)
    if x.shape[1]!=7 or not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(axis=1),1,atol=1e-4):
        raise ValueError('Expected seven finite non-negative probabilities summing to one.')
    p=transform(x,spec['class_offsets'],spec['temperature']);dest=p.argmax(1);cuts=np.array([*spec['queue_cutoffs'],1.01]);auto=(dest<6)&(p.max(1)>=cuts[dest])
    return p,dest,auto
def fit(private,out):
    raw=np.load(private);dates=raw['date'];adapt=dates<'2026-06-08';select=(dates>='2026-06-08')&(dates<'2026-06-12');audit=(dates>='2026-06-12')&(dates<'2026-06-16')
    # NB no June 16–30 or July outcomes used to fit, tune or certify. Previous month's
    # aggregate demonstration motivates the change; this is not a blind June experiment.
    x=raw['p_legacy'];Y=raw['y'];b,t=calibrate(x[adapt],Y[adapt]);p=transform(x,b,t)
    cuts=candidates(p[select],Y[select]);rows=certify(p[audit],Y[audit],cuts,True,alpha=.025)
    gc=candidates(p[select],Y[select],False);grow=certify(p[audit],Y[audit],gc,False,alpha=.025)[0]
    spec={'version':VERSION,'labels':LABELS,'target':RISK,'alpha':.025,'fit_period':'2026-06-01/07','threshold_period':'2026-06-08/11','audit_period':'2026-06-12/15','counts':dict(fit=int(adapt.sum()),tune=int(select.sum()),certify=int(audit.sum())),'class_offsets':b,'temperature':t,'queue_cutoffs':[r['cutoff'] for r in rows],'certificates':rows,'pooled_cutoff':grow['cutoff'],'pooled_certificate':grow,'legacy_cutoff':.9000000000000004,'policy_notes':'Six fixed candidate rules tested together using exact Holm step-down at 2.5% per look. A 1.01 cutoff disables a destination. No coverage claim applies outside eligible narrations with trusted tenant. Statistical replay assumes exchangeable independent cases within the certification window; subsequent shifts are not covered.'}
    # Separate end-of-month reassessment: no threshold selection or refitting after June 11.
    # Short-window no-go is preserved. End-of-month scope was selected after seeing the
    # initial June result, before opening July labels. Report this transparently.
    (out/'june_midmonth_policy.json').write_text(json.dumps(spec,indent=2)+'\n')
    mature=dates>='2026-06-12'
    rows2=certify(p[mature],Y[mature],cuts,True,alpha=.025)
    grow2=certify(p[mature],Y[mature],gc,False,alpha=.025)[0]
    # Using .025 at both looks covers 2 opportunities to release each Holm-controlled family.
    spec['pilot_audit_rule']='End-of-month safety reassessment; fixed thresholds, one-sided Holm .025 in 6 destinations. Pooled gate is a separate benchmark. Not a prospective test of the study design: initial short-window limitation prompted the re-audit.'
    spec['audit_period']='2026-06-12/30';spec['alpha']=.025;spec['counts']['certify']=int(mature.sum())
    spec['queue_cutoffs']=[r['cutoff'] for r in rows2];spec['certificates']=rows2;spec['pooled_cutoff']=grow2['cutoff'];spec['pooled_certificate']=grow2
    (out/'locked_policy.json').write_text(json.dumps(spec,indent=2)+'\n')
    return spec
def measure(x,y,dates,model,d,m,rows,queue_rows,mat_rows):
    n=len(y);auto=int(m.sum());wrong=int((m&(d!=y)).sum());base=2*auto-25*wrong
    rows.append({'policy':model,'n':n,'auto':auto,'manual':n-auto,'correct_auto':auto-wrong,'wrong':wrong,'coverage':auto/n,'risk':wrong/auto if auto else None,'coverage_ci':ci(auto,n),'risk_ci':ci(wrong,auto),'net_scenario_per_10000':base/n*10000})
    for j,name in enumerate(LABELS):
        m0=m&(d==j);A=int(m0.sum());W=int((m0&(d!=y)).sum())
        queue_rows.append(dict(policy=model,destination=name,auto=A,wrong=W,risk=W/A if A else None,risk_upper=ci(W,A)[1],n_actual=int((y==j).sum()),recall_all=float((d[y==j]==j).mean()) if (y==j).any() else None,coverage_of_actual=float(m[y==j].mean()) if (y==j).any() else None))
    for j in range(7):
        for h in range(7):
            mat_rows.append(dict(policy=model,truth=LABELS[j],destination=LABELS[h],n=int((m&(d==h)&(y==j)).sum())))
def evaluate(raw,label,spec,out,name,clamp_dates=None):
    if label is None:idate,x,y=raw['date'],raw['p_legacy'],raw['y']
    else:
        assert np.array_equal(raw['id'],label['id']);idate,x,y=raw['date'],raw['p_legacy'],label['y']
    if clamp_dates is not None:
        sel=idate>=clamp_dates;idate,x,y=idate[sel],x[sel],y[sel]
    p,new,guard=apply(x,spec);prior=x.argmax(1);L=(prior<6)&(x.max(1)>=spec['legacy_cutoff']);P=(new<6)&(p.max(1)>=spec['pooled_cutoff'])
    methods={'manual':(prior,np.zeros(len(x),bool)),'legacy':(prior,L),'recalibrated_same_cutoff':(new,(new<6)&(p.max(1)>=spec['legacy_cutoff'])),'recalibrated_pooled_certificate':(new,P),'certified_destinations':(new,guard)}
    rows=[];qs=[];ms=[]
    for key,(d,m) in methods.items():measure(x,y,idate,key,d,m,rows,qs,ms)
    rng=np.random.default_rng(15071);dates=np.unique(idate)
    days=[np.flatnonzero(idate==k) for k in dates];arr=[]
    for inds in days:
        n=len(inds);lg=int(L[inds].sum());lgw=int(((prior[inds]!=y[inds])&L[inds]).sum());gg=int(guard[inds].sum());ggw=int(((new[inds]!=y[inds])&guard[inds]).sum());arr.append([n,lg,lgw,gg,ggw])
    arr=np.array(arr);draws=arr[rng.integers(0,len(days),size=(3000,len(days)))].sum(axis=1);N,l,lw,g,gw=draws.T
    def bounds(v):return np.quantile(v[np.isfinite(v)],[.025,.975]).tolist() if np.isfinite(v).any() else [None,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        cluster={'days':len(days),'draws':3000,'guard_risk':bounds(gw/g),'legacy_risk':bounds(lw/l),'guard_minus_legacy_risk':bounds(gw/g-lw/l),'guard_minus_legacy_coverage':bounds((g-l)/N),'guard_minus_legacy_net_per_10000':bounds(((g-l)*2-(gw-lw)*25)/N*10000)}
    (out/(name+'_metrics.json')).write_text(json.dumps({'rows':rows,'paired_date_bootstrap':cluster},indent=2)+'\n')
    pd.DataFrame(qs).to_csv(out/(name+'_queues.csv'),index=False);pd.DataFrame(ms).to_csv(out/(name+'_confusion.csv'),index=False)
    sens=[]
    for v in rows:
        for mins in [2,4,7]:
            for cost in [10,25,50]:
                sens.append(dict(policy=v['policy'],review_minutes=mins,loaded_hourly_rate=30,incremental_wrong_route_cost=cost,scenario_net_per_10000=10000*(mins*.5*v['auto']-cost*v['wrong'])/len(y)))
    pd.DataFrame(sens).to_csv(out/(name+'_costs.csv'),index=False)
    return rows
def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=Path(__file__).parent); a.add_argument('--june',type=Path); a.add_argument('--july-predictions',type=Path);a.add_argument('--july-labels',type=Path)
    v=a.parse_args();out=v.out
    if v.june:
        fit(v.june,out);spec=json.loads((out/'june_midmonth_policy.json').read_text());evaluate(np.load(str(v.june)),None,spec,out,'june_late','2026-06-16')
    if v.july_predictions:
        assert v.july_labels
        s=json.loads((out/'locked_policy.json').read_text());evaluate(np.load(str(v.july_predictions)),np.load(str(v.july_labels)),s,out,'july')
if __name__=='__main__':main()
