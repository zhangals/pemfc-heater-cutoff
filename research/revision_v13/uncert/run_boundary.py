"""Run prescribed-boundary perturbations at fixed heater commands."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/'research/study_v2'))
import engine as eng
import numpy as np
from scipy import sparse
from 模型物理 import FuelCellModel, build_shared

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=eng.encode)+'\n',encoding='utf8')

def build(job):
    base,ti,ta,_=eng.make_model({'refine':job['r'],'overrides':{'h':job['h_W_m2K']}})
    def changed(*args):
        m=build_shared(*args)
        g=m.conductance_W_K.copy()
        edge=np.asarray(((m.kind[:-1]=='EP')&(m.kind[1:]=='BP'))|((m.kind[:-1]=='BP')&(m.kind[1:]=='EP')))
        assert edge.sum()==2
        g[edge]=1/(1/g[edge]+job['Rc_m2K_W']/m.area_m2)
        C=m.capacity_J_K.copy();rhocp=m.rhocp.copy()
        C[m.kind=='EP']*=job['cap_factor'];rhocp[m.kind=='EP']*=job['cap_factor']
        outward=m.area_m2*job['h_W_m2K']+np.array([job['left_fixture_W_K'],0.])
        boundary=1/(1/outward+m.dx[[0,-1]]/(2*m.k[[0,-1]]*m.area_m2))
        diag=np.r_[g,0]+np.r_[0,g];diag[[0,-1]]+=boundary
        K=sparse.diags((-g,diag,-g),(-1,0,1),format='csr')
        assert np.max(abs(np.asarray(K.sum(axis=1)).ravel()-np.pad(boundary[:1],(0,len(C)-1))-np.pad(boundary[1:],(len(C)-1,0))))<1e-8
        return replace(m,capacity_J_K=C,rhocp=rhocp,conductance_W_K=g,boundary_conductance_W_K=boundary,K_W_K=K,A_s_inv=(-sparse.diags(1/C)@K).tocsr())
    return FuelCellModel(n_cells=5,refine=job['r'],endplates=True,config=base.cfg,backend='numba',mesh_builder=changed),ti,ta

def run(job_out):
    job, dest = job_out
    key=job['scenario']+'_'+job['command']
    dest.mkdir(parents=True,exist_ok=True)
    record=dest/(key+'.json')
    if record.is_file():return
    model,ti,ta=build(job)
    q,_,knots,_=eng.waveform([job['q_W_cm2']]*5,job['tau_s'],'constant')
    raw,final,segs,_=eng.integrate(model,180.,ti,ta,eng.current_ramp,q,breakpoints=knots,dense_interval=[120,180])
    eng.add_accounting(raw,model,model.initial(ti),final,eng.current_ramp)
    window=eng.analyze_dense(segs,[120,180],raw['time_s'],threshold=1)
    audit=eng.solver.validate_constraints(raw,require_success=False)
    t=np.asarray(raw['series']['time_s']);T=np.asarray(raw['states_scaled'])[:,model.sl['T']]*model.scale[model.sl['T']]
    storage={}
    # Common 38–55.5 s comparison is post-cutoff throughout only for the early command.
    if job['command']=='early':
        i=[int(np.flatnonzero(np.isclose(t,at,rtol=0,atol=1e-8))[0]) for at in (38.,55.5)]
        dT=T[i[1]]-T[i[0]]
        ep=model.mesh.kind=='EP'
        storage={'storage_EP_J':float(dT[ep]@model.mesh.capacity_J_K[ep]),'storage_stack_J':float(dT@model.mesh.capacity_J_K),
                 'signed_min_MEA_change_K':float(np.min(model.mesh.mea_means(T[i[1]]))-np.min(model.mesh.mea_means(T[i[0]])))}
    data=dict(job,status=raw['status'],reason=raw.get('failure_reason'),time_s=raw['time_s'],
              minimum_C=window['minimum_mea_mean_C'],cell=window['minimum_cell_one_based'],time_min_s=window['minimum_time_s'],
              nominal_pass=(raw['status']=='COMPLETE' and window['valid'] and audit['passed'] and window['minimum_mea_mean_C']>=1),energy_J=raw['aux_energy_J'],audit=audit,**storage)
    np.savez_compressed(dest/(key+'.npz'),time_s=t,mea_C=model.mesh.mea_means(T)-273.15,endplate_sensible_J=(T[:,model.mesh.kind=='EP']-273.15)@model.mesh.capacity_J_K[model.mesh.kind=='EP'],dense_times=np.array([[s['t0'],s['t1']] for s in segs]),dense_orders=np.array([s['order'] for s in segs]),dense_coef=np.array([np.pad(s['coefficients'],((0,6-len(s['coefficients'])),(0,0))) for s in segs]))
    save(record,data)
    print(key,raw['status'],data['minimum_C'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Fresh output directory')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--scenario',choices=['ref','low_h','high_h','contact','fixture','combined'])
    p.add_argument('--command',choices=['early','extended','table1','double'])
    args=p.parse_args()
    jobs=read(ROOT/'results/v13/boundaries/protocol.json')['jobs']
    selected=[(job,args.output) for job in jobs if args.scenario in (None,job['scenario']) and args.command in (None,job['command'])]
    args.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(args.workers) as pool:list(pool.map(run,selected))
