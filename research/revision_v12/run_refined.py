"""Run the ten fixed-control five-cell tests on the refine=8 mesh."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/study_v2'))
import engine as eng
import numpy as np


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=eng.encode)+'\n',encoding='utf-8')


def solve(args):
    job,folder = args
    output=folder/job['name']
    output.mkdir(exist_ok=False)
    model,ti,ta,_=eng.make_model({'refine':8,'overrides':{}})
    command,dose,knots,_=eng.waveform(job['powers'],job['cutoff_s'],job['profile'])
    run,final,segments,_=eng.integrate(model,180.,ti,ta,eng.current_ramp,command,
                                       breakpoints=knots,dense_interval=[120.,180.])
    eng.add_accounting(run,model,model.initial(ti),final,eng.current_ramp)
    late=eng.analyze_dense(segments,[120.,180.],run['time_s'],threshold=job['target_C'])
    audit=eng.solver.validate_constraints(run,require_success=False)
    complete=run['status']=='COMPLETE' and abs(run['time_s']-180.)<1e-8 and late['valid']
    low=late['minimum_mea_mean_C'] if complete else None
    dose_error=float(np.max(np.abs(25*dose(run['time_s'])-np.asarray(run['aux_energy_cells_J']))))
    result={**job,'mesh':8,'integrated_to_s':run['time_s'],'status':run['status'],
            'energy_J':run['aux_energy_J'],'audit':audit,'maximum_cell_dose_error_J':dose_error,
            'thermal_nodes':model.nt,'state_count':model.size,'minimum_C':low,
            'cell':late['minimum_cell_one_based'],'active_time_s':late['minimum_time_s'],
            'passes':bool(complete and audit['passed'] and dose_error<1e-4 and low>=job['target_C'])}
    save(output/'result.json',result)
    times,coefficients,orders=[],[],[]
    for segment in segments:
        times.append([segment['t0'],segment['t1']])
        orders.append(segment['order'])
        values=segment['coefficients']
        block=np.zeros((6,5))
        block[:len(values)]=values
        coefficients.append(block)
    np.savez_compressed(output/'trajectory.npz',time_s=run['series']['time_s'],
                        mea_C=run['series']['temperature_C'],voltage_V=run['series']['voltage_V'],
                        dense_times_s=times,properties=coefficients,orders=orders)
    print(job['name'],run['status'],low,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('reruns/refine8'))
    ap.add_argument('--case',help='Run one named case from cases.json')
    ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args()
    jobs=json.loads((ROOT/'results/review_checks/cases.json').read_text(encoding='utf-8'))
    if args.case is not None:
        jobs=[j for j in jobs if j['name']==args.case]
        if not jobs:ap.error('Unknown case')
    target=args.output.resolve()
    target.mkdir(parents=True,exist_ok=True)
    if any((target/j['name']).exists() for j in jobs):raise FileExistsError('Use a fresh output directory.')
    if args.workers==1:
        results=[solve((j,target)) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(solve,[(j,target) for j in jobs]))
    save(target/'summary.json',results)


if __name__=='__main__':main()

