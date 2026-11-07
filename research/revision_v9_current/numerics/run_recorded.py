"""Recompute single-cell comparisons with the recorded current-density column.

The sole source of simulation inputs is the original competition workbook.
Run this script from the workspace root. Each folder/run is created once.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
import hashlib
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
V3 = ROOT / 'research/study_v3_diagnostics/cross_condition'
sys.path.insert(0, str(V3))
import run_cross_condition as original
import numpy as np
import openpyxl
from scipy.optimize import least_squares
from 模型物理 import FROZEN_CONFIG

BOOK = ROOT / 'local_inputs/attachment_2.xlsx'

TEMPS = (-20, -25)
PARAMS = original.NAMES
BUDGET = 48

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, content):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(content,handle,ensure_ascii=False,indent=2,allow_nan=False,default=original.eng.encode)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def records():
    book = openpyxl.load_workbook(BOOK, read_only=True, data_only=True)
    result = {}
    for t in TEMPS:
        rows = list(book[f'{t}℃'].values)
        assert '电流密度' in rows[1][4] and len(rows)==186
        a = np.asarray(rows[2:],dtype=float)
        assert a.shape == (184,5) and np.isfinite(a).all()
        assert np.allclose(np.diff(a[:,0]),.2,rtol=0,atol=1e-10)
        result[t] = np.column_stack([np.full(184,t),a])
    book.close()
    return result

def freeze():
    files = [Path(__file__),BOOK,V3/'run_cross_condition.py',V3/'protocol.json',
             ROOT/'research/study_v2/engine.py',ROOT/'代码/模型物理.py',ROOT/'代码/数值求解.py']
    old = read(V3/'protocol.json')
    write(OUT/'protocol.json',{
        'utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': {p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'raw_input':'Attachment 2, worksheets −20 and −25 °C, original column E in A cm^-2. Use linear interpolation at original times. Each sheet contains 184 samples, 0 to 36.6 s.',
        'training_temperatures_C':TEMPS,
        'names':PARAMS,'bounds':[original.LOW,original.HIGH],'starts':old['starts'],
        'training_mesh':2,'integrations_per_start_limit':BUDGET,'partial_trajectory_penalty':old['incomplete_candidate'],
        'objective':old['objective'],'optimizer':old['optimizer'],'selection':old['parameter_selection'],
        'geometry':old['geometry'],'numerics':old['numerics'],'assessment_meshes':[2,4],
        'nominal_parameters':{n:FROZEN_CONFIG[n] for n in PARAMS},
        'assessment':'For each independently selected training-temperature fit, run both temperatures at both meshes. Separately run both temperatures and meshes using the prescribed nominal parameter values. Both training and evaluation use recorded column E.',
        'interpretation':old['interpretation']})

def fresh_solve(folder, p, data, refine, mode, extras):
    folder.mkdir()
    residual,pred,meta,raw,final,model=original.simulate(p,data,refine=refine,convention='recorded',fields=False)
    assert meta['convention']=='recorded'
    meta.update(mode=mode,**extras)
    np.savez_compressed(folder/'trajectory.npz',
        prediction_at_observation_times=pred if pred is not None else np.empty((0,2)),
        times_s=raw['series']['time_s'],voltage_V=raw['series']['voltage_V'],
        temperature_C=raw['series']['temperature_C'],final_scaled=final,parameters=p)
    write(folder/'result.json',meta)
    return residual,meta

def train(temp,start):
    data=records()[temp]
    parent=OUT/f'train_m{abs(temp)}_start{start+1}'
    parent.mkdir()
    p0=read(OUT/'protocol.json')['starts'][start]
    trials=[]
    class BudgetReached(Exception):
        pass
    def objective(x):
        if len(trials)>=BUDGET:
            raise BudgetReached
        number=len(trials)+1
        residual,meta=fresh_solve(parent/f'eval_{number:03d}',original.decode(x),data,2,'training',
            {'training_temperature_C':temp,'start':start+1,'evaluation':number})
        trials.append(meta)
        if residual is None:
            return np.full(368,np.sqrt((10000+100*(36.6-meta['time_s']))/368))
        return residual
    try:
        least_squares(objective, original.encode(np.asarray(p0)),bounds=(np.zeros(7),np.ones(7)),
                      method='trf',diff_step=1e-3,max_nfev=48)
    except BudgetReached:
        pass
    complete=[trial for trial in trials if trial['complete']]
    best=min(complete,key=lambda trial:trial['metrics']['loss']) if complete else None
    info={'temperature_C':temp,'start':start+1,'training_calls':len(trials),'complete':len(complete),'selected':best}
    write(parent/'selection.json',info)
    return info

def assess(target,mesh,train_temp=None,p=None):
    if train_temp is None:
        mode='nominal'
        name=f'nominal_m{abs(target)}_r{mesh}'
        p=np.array([FROZEN_CONFIG[n] for n in PARAMS])
    else:
        mode='fit' if target==train_temp else 'cross_condition'
        name=f'assess_train_m{abs(train_temp)}_target_m{abs(target)}_r{mesh}'
    _,meta=fresh_solve(OUT/name,p,records()[target],mesh,mode,{'training_temperature_C':train_temp,'target_temperature_C':target})
    return {'trajectory_folder':name,**meta}

def main():
    freeze()
    assessments=[]
    selections={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        nominal=[pool.submit(assess,t,g) for t in TEMPS for g in (2,4)]
        fits=[pool.submit(train,t,k) for t in TEMPS for k in range(2)]
        for future in as_completed(nominal):
            assessments.append(future.result())
        write(OUT/'nominal_metrics.json',assessments)
        print('4 nominal solves complete.',flush=True)
        by_temp={t:[] for t in TEMPS}
        for future in as_completed(fits):
            run=future.result()
            by_temp[run['temperature_C']].append(run)
            print('Fit complete',run['temperature_C'],run['start'],run['training_calls'],flush=True)
        for temp in TEMPS:
            candidates=[r['selected'] for r in by_temp[temp] if r['selected'] is not None]
            selections[temp]=min(candidates,key=lambda r:(r['metrics']['loss'],r['start'],r['evaluation']))
        write(OUT/'selected_parameters.json',selections)
        predictions=[]
        for train_temp,model in selections.items():
            p=np.asarray([model['parameters'][name] for name in PARAMS])
            predictions.extend(pool.submit(assess,target,refine,train_temp,p) for target in TEMPS for refine in (2,4))
        for future in as_completed(predictions):
            assessments.append(future.result())
    result={'protocol_sha256':sha(OUT/'protocol.json'),'input_column':'Attachment 2 column E',
            'training_counts':{t:sum(r['training_calls'] for r in runs) for t,runs in by_temp.items()},
            'training_complete_counts':{t:sum(r['complete'] for r in runs) for t,runs in by_temp.items()},
            'selected_fits':selections,'assessments':sorted(assessments,key=lambda a:a['trajectory_folder'])}
    assert len(assessments)==12
    write(OUT/'metrics.json',result)
    print('Saved final metrics',flush=True)
if __name__=='__main__':
    main()

