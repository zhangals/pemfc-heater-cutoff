"""Fresh fixed-power cutoff scan testing recovery after structural heat-capacity changes."""
from pathlib import Path
import json
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
import time
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'research/study_v2'))
import engine as eng
import run_structural_phase_diagnostics as structural
from transition_diagnostics import interval_diagnostic
import numpy as np
OUT=BASE/'design_recovery'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def new(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=eng.encode,allow_nan=False)
def sources():
 d=structural.source_hashes();d[eng.relative(Path(__file__))]=eng.digest(Path(__file__));return d
def freeze():
 jobs=[dict(id=f'capacity{str(factor).replace(".","p")}_tau{tau}_r{r}',factor=factor,cutoff=tau,refine=r)
  for factor in [.5,1.,2.] for tau in [20,40,60,80,90] for r in [2,4]]
 new(OUT/'protocol.json',dict(frozen_utc=eng.utc_now(),sources=sources(),jobs=jobs,
  question='Can extending a bounded uniform heater pulse restore the common late-window requirement after a change in endplate heat capacity?',
  controls='All cells 1 W/cm2, cutoff from frozen discrete grid 20/40/60/80/90 s; same ramp, -30C and 120-180s >=1C criterion.',
  analysis='Report all 30 grid points, complete versus domain termination and actual stopping times. A successful longer pulse is a design recovery, not global energy optimality.',
  selection='Smallest sampled cutoff satisfying the requirement on both meshes; if none, report unresolved within this tested family.',
  numerics=eng.PROTOCOL['numerics'],prescribed_parameter_factors_not_measured_uncertainty=True))
def run(job):
 recipe=read(OUT/'protocol.json')
 if recipe['sources']!=sources():raise ValueError('Frozen sources changed')
 path=OUT/'runs'/(job['id']+'.json');archive=path.with_suffix('.npz');path.parent.mkdir(exist_ok=True)
 if path.exists() or archive.exists():raise FileExistsError(job['id'])
 start=time.perf_counter()
 sj={'refine':job['refine'],'scenario':{'id':job['id'],'ep_capacity_factor':job['factor']}}
 model,initial,ti,ta=structural.model_for(sj)
 command,_,knots,_=eng.waveform([1.]*5,job['cutoff'],'constant')
 raw,final,segments,sol=eng.integrate(model,180.,ti,ta,eng.current_ramp,command,breakpoints=knots,initial_state=initial,dense_interval=[0.,180.])
 eng.add_accounting(raw,model,initial,final,eng.current_ramp)
 sampled,series,physical=structural.sampled_diagnostics(model,raw,initial,final,job['cutoff'])
 arrays=eng.model_arrays(model);arrays.update(eng.archive_stage(raw,final,initial,segments));arrays.update(series,sampled_states_physical=physical)
 dense={}
 for name,interval in [('postcutoff_to120',[job['cutoff'],120.]),('late_window',[120.,180.])]:
  dense[name]=interval_diagnostic(arrays['dense_segment_times_s'],arrays['dense_temperature_coefficients_C'],arrays['dense_orders'],interval) if raw['time_s']>=interval[1]-1e-8 else {'full_dense_coverage':False,'minimum_mea_mean_C':None}
 raw.update(job_id=job['id'],job=job,sources=sources(),recipe_sha256=eng.digest(OUT/'protocol.json'),
  fresh_integration=True,reused_prior_result=False,complete_180s=raw['status']=='COMPLETE' and raw['time_s']>=180.-1e-8,
  solver_kwargs=sol,sampled_diagnostics=sampled,dense_diagnostics=dense)
 eng.finish(path,archive,raw,arrays,start)
 return {k:raw.get(k) for k in ['job_id','status','time_s','aux_energy_J','dense_diagnostics']}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 if a.freeze:freeze()
 else:
  jobs=read(OUT/'protocol.json')['jobs']
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   futures=[pool.submit(run,j) for j in jobs]
   for f in as_completed(futures):print(json.dumps(f.result(),default=eng.encode),flush=True)
