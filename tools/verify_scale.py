"""Audit role counts and recompute rates from independent aggregate tables."""
import csv, json, math
from pathlib import Path
def verify(root):
    folder=root/'experiments/scale/results'
    rows=lambda n:list(csv.DictReader((folder/(n+'.csv')).read_text().splitlines()))
    read=lambda n:json.loads((folder/(n+'.json')).read_text())
    s=read('selection'); f=read('fit')
    assert f['original_training']+s['train_added']['keep']==f['expanded_training']==54008
    assert s['train_added']['keep']+s['test_new']['keep']+s['may_additional']['keep']==51817
    for name in ['gates','rank']:
        a,b,c=[rows(part+'_'+name) for part in ['original','additional','all']]
        for x,y,z in zip(a,b,c,strict=True):
            for k in (['group_n','routed','wrong'] if name=='gates' else ['n','correct_in_list']):
                assert int(x[k])+int(y[k])==int(z[k])
    for v in rows('learning'):
        assert math.isclose(float(v['new_f1'])-float(v['old_f1']),float(v['delta']),abs_tol=1e-12)
        assert float(v['lo'])<=float(v['delta'])<=float(v['hi'])
    assert sum(int(r['n']) for r in rows('months') if r['test']=='additional historical test')==s['test_new']['keep']
if __name__=='__main__':
    verify(Path(__file__).resolve().parents[1]);print('Verified expanded cohorts, disjoint May counts, and model comparison aggregates.')
