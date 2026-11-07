"""Four predeclared bisection rounds of each observed fixed-power recovery bracket."""
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import run_design_recovery as d

BASE=Path(__file__).resolve().parent
OUT=BASE/'design_refinement'
original_sources=d.sources
def sources():
    s=original_sources();s[d.eng.relative(Path(__file__))]=d.eng.digest(Path(__file__));return s
def setup():
    d.OUT=OUT;d.sources=sources
def run(job):
    setup();return d.run(job)
def good(rows):
    return all(r['status']=='COMPLETE' and r['dense_diagnostics']['late_window']['minimum_mea_mean_C']>=1. for r in rows)
def main():
    setup()
    brackets={.5:[20.,40.],1.:[40.,60.],2.:[80.,90.]}
    initial={}
    for f,ab in brackets.items():
        for t in ab:
            paths=[BASE/'design_recovery/runs'/f'capacity{str(f).replace(".","p")}_tau{int(t)}_r{r}.json' for r in (2,4)]
            initial[f'{f}:{t}']=[{'path':d.eng.relative(p),'sha256':d.eng.digest(p)} for p in paths]
            assert good([d.read(p) for p in paths])==(t==ab[1])
    d.new(OUT/'protocol.json',dict(frozen_utc=d.eng.utc_now(),sources=sources(),initial_brackets=brackets,
        initial_records=initial,rounds=4,integrations=24,criterion='Both r2 and r4 complete 180 s and dense 120-180s minimum >=1 C.',
        rule='Evaluate midpoint on both meshes; replace upper if both pass, otherwise lower. Record all failures. Report sampled passing cutoff and bracketing interval, not a proof of monotonicity or global optimum.',
        numerics=d.eng.PROTOCOL['numerics'],controls='Uniform 1 W/cm2 on five cells; all other physical inputs unchanged.'))
    history=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for n in range(1,5):
            jobs=[dict(id=f'cap{str(f).replace(".","p")}_round{n}_r{r}',factor=f,cutoff=sum(ab)/2,refine=r) for f,ab in brackets.items() for r in (2,4)]
            rows=list(pool.map(run,jobs))
            for f,ab in brackets.items():
                pair=[r for r,j in zip(rows,jobs) if j['factor']==f]
                mid=sum(ab)/2;passed=good(pair)
                history.append(dict(round=n,factor=f,cutoff=mid,passed=passed,results=pair))
                ab[1 if passed else 0]=mid
                print(json.dumps(dict(round=n,factor=f,cutoff=mid,passed=passed,bracket=ab)),flush=True)
    d.new(OUT/'summary.json',dict(protocol_sha256=d.eng.digest(OUT/'protocol.json'),brackets=brackets,history=history,complete=True))
if __name__=='__main__':main()
