"""Propagate two saved coefficient vectors through two fixed stack schedules."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from run_boundary import ROOT, build, eng

def solve(spec):
    command,fit_C,job,fit,out=spec
    nominal,ti,ta=build(job)
    model=eng.FuelCellModel(n_cells=5,refine=4,endplates=True,
         config=replace(nominal.cfg,**fit),backend='numba',mesh_builder=lambda *args:nominal.mesh)
    assert len(fit)==7 and all(getattr(model.cfg,k)==v for k,v in fit.items())
    q,_,knots,_=eng.waveform([job['q_W_cm2']]*5,job['tau_s'],'constant')
    raw,final,segments,_=eng.integrate(model,180.,ti,ta,eng.current_ramp,q,breakpoints=knots,dense_interval=[120.,180.])
    eng.add_accounting(raw,model,model.initial(ti),final,eng.current_ramp)
    result=eng.analyze_dense(segments,[120.,180.],raw['time_s'],threshold=1)
    audit=eng.solver.validate_constraints(raw,require_success=False)
    filename=f'{command}_fit{abs(fit_C)}'
    dest=out/(filename+'.npz')
    np.savez_compressed(dest,time_s=raw['series']['time_s'],mea_C=raw['series']['temperature_C'],
       times=np.array([[s['t0'],s['t1']] for s in segments]),orders=np.array([s['order'] for s in segments]),
       coef=np.array([np.pad(s['coefficients'],((0,6-len(s['coefficients'])),(0,0))) for s in segments]))
    row=dict(command=command,fit_C=fit_C,cutoff_s=job['tau_s'],power_W_cm2=job['q_W_cm2'],endplate_capacity_factor=job['cap_factor'],
        status=raw['status'],reason=raw.get('failure_reason'),end_s=raw['time_s'],energy_J=raw['aux_energy_J'],
        minimum_C=result['minimum_mea_mean_C'],mesh=4,full_window=result['valid'],constraint_audit=audit)
    (out/(filename+'.json')).write_text(json.dumps(row,indent=2,default=eng.encode,allow_nan=False),encoding='utf8')
    print(filename,row['status'],row['minimum_C'],flush=True)

def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--protocol',type=Path,default=ROOT/'results/v13/boundaries/protocol.json')
    p.add_argument('--command',choices=['table1','double']);p.add_argument('--fit',type=int,choices=[-20,-25])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    fits=json.loads((ROOT/'research/revision_v9_current/numerics/selected_parameters.json').read_text(encoding='utf8'))
    jobs=json.loads(args.protocol.read_text(encoding='utf8'))['jobs']
    specs=[(j['command'],fit,j,fits[str(fit)]['parameters'],args.output) for j in jobs if j['command'] in ('table1','double') and j['scenario']=='ref' for fit in (-20,-25) if args.command in (None,j['command']) and args.fit in (None,fit)]
    with ProcessPoolExecutor(args.workers) as pool:list(pool.map(solve,specs))
if __name__=='__main__':main()
