"""Fixed-total-power regional heater comparisons for the nominal five-cell stack."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
V2 = ROOT / 'research/study_v2'
sys.path.insert(0,str(V2))
import engine as eng
import numpy as np

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def new(path,data):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(data,handle,ensure_ascii=False,indent=2,default=eng.encode,allow_nan=False)

def freeze():
    rep = next(r for r in read(V2/'representatives.json')['representatives']
               if r['family']=='uniform' and r['reserve_C']==1)
    q, cutoff = rep['design_x']
    directions={'uniform':[0,0,0,0,0],'edge_inner_plus':[1,-1,0,-1,1],
                'edge_inner_minus':[-1,1,0,1,-1],'edge_center_plus':[1,0,-2,0,1],
                'edge_center_minus':[-1,0,2,0,-1]}
    powers={name:(q+.10*np.array(offsets)).tolist() for name,offsets in directions.items()}
    assert all(min(v)>=0 and max(v)<=1 and abs(sum(v)-5*q)<1e-12 for v in powers.values())
    new(OUT/'protocol.json',{'frozen_utc':eng.utc_now(),'parent':rep,'reference_q_W_cm2':q,
        'cutoff_s':cutoff,'delta_W_cm2':.10,'cell_order':'end, inner, central, inner, end',
        'powers':powers,'meshes':[2,4],'initial_and_load':'Fresh nominal dry −30 C state, common 0.005*t load ramp capped at 0.3 A/cm2',
        'reserve_and_horizon':'1 C in the dense window from 120 to 180 s, all five cells, 0.30 V floor and original model domain constraints',
        'cost':'25 cm2 times sum of commanded densities times heater duration. Record integrated delivered heat separately.',
        'temperature_difference':'Mean of two end-cell MEA means minus center-cell MEA mean, in K, at cutoff, 120 s, late thermal minimum and 180 s.',
        'shortening_rule':'Eligible perturbations improve dense-window minima relative to the uniform base by >1e-4 K on both meshes and satisfy the full protocol on both. Select the highest worst-mesh minimum, tie by name. Test offsets 0.125/0.5/1/2 s shorter at fixed densities on both meshes; at the shortest paired pass (if none, the smallest offset), repeat uniform at the same cutoff on both meshes. Complete this finite comparison irrespective of outcome.',
        'sources':dict(eng.source_hashes(),**{eng.relative(p):eng.digest(p) for p in [Path(__file__),V2/'representatives.json']})})

def dense_values(segments,time_s):
    for seg in segments:
        if seg['t0']-1e-9<=time_s<=seg['t1']+1e-9:
            z=np.clip((time_s-seg['t0'])/(seg['t1']-seg['t0']),0,1)
            return np.polynomial.polynomial.polyval(z,seg['coefficients'])
    return None

def solve(allocation,refine,offset):
    proto=read(OUT/'protocol.json')
    powers=np.array(proto['powers'][allocation])
    cutoff=proto['cutoff_s']-offset
    name=f'{allocation}_r{refine}_minus{round(offset*1000):04d}ms'
    path=OUT/name
    path.mkdir()
    row={'job_id':name,'fresh_integration':True,'started_utc':eng.utc_now(),'allocation':allocation,'refine':refine,
         'cutoff_s':cutoff,'shortening_s':offset,'powers_W_cm2':powers.tolist(),'command_energy_J':25*float(powers.sum())*cutoff,
         'protocol_sha256':eng.digest(OUT/'protocol.json')}
    try:
        model,ti,ta,_=eng.make_model({'refine':refine,'overrides':{}})
        initial=model.initial(ti)
        command,dose,knots,_=eng.waveform(powers,cutoff,'constant')
        raw,final,segments,settings=eng.integrate(model,180.,ti,ta,eng.current_ramp,command,
             breakpoints=knots,dense_interval=[cutoff,180.])
        full=raw['status']=='COMPLETE' and abs(raw['time_s']-180)<1e-8
        eng.add_accounting(raw,model,initial,final,eng.current_ramp)
        audit=eng.solver.validate_constraints(raw,require_success=False)
        dose_error=float(np.max(np.abs(25*dose(raw['time_s'])-np.asarray(raw['aux_energy_cells_J']))))
        late=eng.analyze_dense(segments,[120.,180.],raw['time_s'])
        trans=eng.analyze_dense(segments,[cutoff,120.],raw['time_s'])
        samples=np.array(raw['series']['time_s'])
        celltemps=np.array(raw['series']['temperature_C'])
        tdiff={}
        for label,t in [('cutoff',cutoff),('120s',120.),('late_minimum',late['minimum_time_s']),('180s',180.)]:
            if t is None: continue
            c=dense_values(segments,t)
            if c is not None:
                tdiff[label]={'time_s':t,'end_mean_C':float((c[0]+c[-1])/2),'center_C':float(c[2]),
                              'end_minus_center_K':float((c[0]+c[-1])/2-c[2]),'five_cell_means_C':c.tolist()}
        row.update(status=raw['status'],end_time_s=raw['time_s'],failure_reason=raw.get('failure_reason'),complete=full,
                   passes=bool(full and audit['passed'] and dose_error<=1e-4 and late['valid'] and late['minimum_mea_mean_C']>=1.),
                   constraint_audit=audit,max_cell_dose_error_J=dose_error,late=late,transition=trans,end_center=tdiff,
                   realized_energy_J=raw['aux_energy_J'],electrical_output_J=raw['electrical_output_J'],
                   net_input_J=raw['net_input_energy_J'],min_voltage_V=raw['min_voltage_V'],
                   initial_state_sha256=__import__('hashlib').sha256(initial.tobytes()).hexdigest(),
                   wall_s=raw['wall_s'],time_sample_count=len(samples),solver=settings)
        arrays=eng.model_arrays(model)
        arrays.update(eng.archive_stage(raw,final,initial,segments))
        arrays.update(cell_temperature_C=celltemps,voltage_V=raw['series']['voltage_V'])
        with (path/'trajectory.npz').open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        row['trajectory_sha256']=eng.digest(path/'trajectory.npz')
    except Exception as exc:
        row.update(status='EXCEPTION',complete=False,passes=False,failure_reason=repr(exc),traceback=traceback.format_exc())
    row['finished_utc']=eng.utc_now()
    new(path/'result.json',row)
    print(name,row['status'],row.get('late',{}).get('minimum_mea_mean_C'),flush=True)
    return row

def run_batch(pool,trials):
    return sorted([future.result() for future in as_completed([pool.submit(solve,*job) for job in trials])],
                  key=lambda row:row['job_id'])

def main():
    freeze()
    proto=read(OUT/'protocol.json')
    with ProcessPoolExecutor(max_workers=4) as pool:
        base=run_batch(pool,[(name,r,0.) for name in proto['powers'] for r in (2,4)])
        new(OUT/'matched_energy.json',base)
        lookup={(a['allocation'],a['refine']):a for a in base}
        qualified=[]
        for name in proto['powers']:
            if name=='uniform':continue
            checks=[(lookup[(name,r)],lookup[('uniform',r)]) for r in (2,4)]
            if all(a['passes'] and b['complete'] and a['late']['minimum_mea_mean_C']>b['late']['minimum_mea_mean_C']+1e-4 for a,b in checks):
                qualified.append((min(a['late']['minimum_mea_mean_C'] for a,_ in checks),name))
        winner=sorted(qualified,key=lambda x:(-x[0],x[1]))[0][1] if qualified else None
        new(OUT/'selection.json',{'eligible':qualified,'selected':winner})
        short=[]
        if winner:
            offsets=[.125,.5,1.,2.]
            short=run_batch(pool,[(winner,r,offset) for offset in offsets for r in (2,4)])
            pass_offsets=[o for o in offsets if all(a['passes'] for a in short if a['shortening_s']==o)]
            compare=max(pass_offsets) if pass_offsets else min(offsets)
            short+=run_batch(pool,[('uniform',r,compare) for r in (2,4)])
    new(OUT/'metrics.json',{'protocol_sha256':eng.digest(OUT/'protocol.json'),
        'equal_energy_scenarios':base,'eligible':qualified,'selected_for_shortening':winner,'shortened_cutoffs':short})
if __name__=='__main__':
    main()

