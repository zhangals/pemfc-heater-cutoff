"""Execute the frozen v2 Bayesian design searches without old-run caches."""
from pathlib import Path
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import time
import warnings
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, RBF

BASE = Path(__file__).resolve().parent
PROTOCOL_PATH = BASE/'protocol.json'
P = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
FREEZE = json.loads((BASE/'protocol_freeze.json').read_text(encoding='utf-8'))
PROTOCOL_HASH = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
assert PROTOCOL_HASH == FREEZE['protocol_sha256']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def chain_id(family, reserve, seed):
    return f'{family}_T{reserve:g}_seed{seed}'.replace('.', 'p')


def powers_cutoff(family, x):
    if family == 'uniform': return [float(x[0])]*5, float(x[1])
    return [float(x[0]),float(x[1]),float(x[2]),float(x[1]),float(x[0])],float(x[3])


def known_energy(family, x):
    x = np.asarray(x)
    if family == 'uniform': return 125*x[...,0]*x[...,1]
    return 25*(2*x[...,0]+2*x[...,1]+x[...,2])*x[...,3]


def propose(family, history, pool_unit, seed):
    bounds = np.asarray(P['design_families'][family]['bounds'])
    lo, hi = bounds[:,0], bounds[:,1]
    x = np.asarray([d['design_x'] for d in history]); unit = (x-lo)/(hi-lo)
    distance = np.min(np.max(abs(pool_unit[:,None,:]-unit[None,:,:]),axis=2),axis=1)
    pu = pool_unit[distance>1e-8]
    pool = qmc.scale(pu,lo,hi)
    costs = known_energy(family,pool)
    finite = [i for i,d in enumerate(history) if d['complete_180s'] and d['g_window_K'] is not None]
    classified = [i for i,d in enumerate(history) if d['numerically_resolved']]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if len(finite)>=2:
            kernel = ConstantKernel(1.,(1e-2,1e3))*Matern(np.full(len(lo),.5),(.03,20.),nu=2.5)+WhiteKernel(1e-5,noise_level_bounds='fixed')
            gp = GaussianProcessRegressor(kernel=kernel,alpha=1e-8,normalize_y=True,n_restarts_optimizer=1,random_state=seed)
            gp.fit(unit[finite],[history[i]['g_window_K'] for i in finite])
            mu,sd = gp.predict(pu,return_std=True)
            pg = norm.cdf(mu/np.maximum(sd,1e-10)); kernel_description = str(gp.kernel_)
        else:
            mu=sd=None;pg=np.full(len(pu),.5);kernel_description='Insufficient complete margins; constant probability 0.5.'
        labels=np.asarray([int(history[i]['complete_180s'] and history[i]['audit']['passed']) for i in classified])
        if len(np.unique(labels))==2:
            clf=GaussianProcessClassifier(ConstantKernel(1.,(.1,10.))*RBF(np.full(len(lo),.5),(.03,20.)),n_restarts_optimizer=0,random_state=seed,max_iter_predict=100)
            clf.fit(unit[classified],labels);pc=clf.predict_proba(pu)[:,1];classifier_description=str(clf.kernel_)
        else:
            pc=np.full(len(pu),float((labels.sum()+1)/(len(labels)+2)));classifier_description='Beta(1,1)-smoothed single-class probability.'
        captured=[str(w.message) for w in captured]
    incumbent=min((d['planned_energy_J'] for d in history if d['feasible']),default=None)
    probability=pg*pc
    acq=probability if incumbent is None else np.maximum(incumbent-costs,0)*probability
    fallback=None
    if np.max(acq)<=0: acq=probability;fallback='No positive known-cost improvement; maximum feasibility probability.'
    selected=int(np.argmax(acq))
    info=dict(incumbent_J=incumbent,acquisition=float(acq[selected]),completion_probability=float(pc[selected]),margin_probability=float(pg[selected]),margin_mean_K=None if mu is None else float(mu[selected]),margin_sd_K=None if sd is None else float(sd[selected]),margin_training_indices=finite,completion_training_indices=classified,margin_kernel=kernel_description,completion_kernel=classifier_description,pool_size=len(pool),warnings=captured,fallback=fallback)
    return pool[selected].tolist(),info


def run_chain(family, reserve, seed, resume=False):
    from engine import evaluate_job
    started=time.perf_counter();cid=chain_id(family,reserve,seed)
    chain_path=BASE/'searches'/cid
    bounds=np.asarray(P['design_families'][family]['bounds']);dim=len(bounds)
    initial=[]
    for q,cutoff in P['search']['anchors_uniform_q_and_cutoff']:
        initial.append([q,cutoff] if family=='uniform' else [q,q,q,cutoff])
    initial.extend(qmc.scale(qmc.Sobol(dim,scramble=True,seed=seed).random_base2(2),bounds[:,0],bounds[:,1]).tolist())
    pool_engine=qmc.Sobol(dim,scramble=True,seed=seed+1000)
    history=[];total=P['search']['evaluations_per_family_per_reserve_per_seed']
    for index in range(total):
        if index<len(initial): x=initial[index];acquisition=None
        else:
            pool=pool_engine.random(P['search']['candidate_pool_per_iteration'])
            x,acquisition=propose(family,history,pool,seed)
        jid=f'{cid}_eval{index+1:02d}';q,cutoff=powers_cutoff(family,x)
        job=dict(job_id=jid,phase='search',powers=q,cutoff_s=cutoff,reserve_C=reserve,allowance_K=P['operating_protocol']['coarse_search_numerical_allowance_K'],refine=1,dense=False,profile='constant',family=family,design_x=x,seed=seed,budget_index=index+1,acquisition=acquisition)
        result_path=BASE/'runs'/'search'/f'{jid}.json'
        if result_path.exists():
            if not resume: raise FileExistsError(f'Refusing old result reuse: {result_path}; use explicit v2 --resume only after interruption.')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            assert result.get('frozen_protocol_sha256')==PROTOCOL_HASH
            archived_job=result.get('job',{})
            assert archived_job.get('job_id')==jid and np.allclose(archived_job['powers'],q,rtol=0,atol=1e-12) and abs(archived_job['cutoff_s']-cutoff)<1e-12
        else: result=evaluate_job(job)
        # These descriptors duplicate the requested job, not any inferred result.
        result['design_x']=x;result['family']=family;result['seed']=seed;result['budget_index']=index+1
        history.append(result)
        feasible=[d for d in history if d['feasible']]
        best=min(feasible,key=lambda d:d['planned_energy_J']) if feasible else None
        write_json(chain_path/'progress.json',dict(chain_id=cid,completed=len(history),budget=total,feasible_count=len(feasible),best_id=None if best is None else best['job']['job_id'],best_energy_J=None if best is None else best['planned_energy_J'],elapsed_s=time.perf_counter()-started,protocol_sha256=PROTOCOL_HASH))
        if acquisition is not None: write_json(chain_path/f'acquisition_{index+1:02d}.json',acquisition)
        print(json.dumps(dict(chain=cid,completed=len(history),status=result['status'],feasible=result['feasible'],best_J=None if best is None else best['planned_energy_J'])),flush=True)
    winner=min((d for d in history if d['feasible']),key=lambda d:d['planned_energy_J'],default=None)
    fields=['job','design_x','planned_energy_J','actual_aux_energy_J','g_window_K','feasible','status']
    summary=dict(chain_id=cid,family=family,reserve_C=reserve,seed=seed,evaluations=len(history),feasible_count=sum(d['feasible'] for d in history),numerically_unresolved=sum(not d['numerically_resolved'] for d in history),winner=None if winner is None else {k:winner[k] for k in fields},job_ids=[d['job']['job_id'] for d in history],protocol_sha256=PROTOCOL_HASH,elapsed_s=time.perf_counter()-started)
    write_json(chain_path/'summary.json',summary)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--resume',action='store_true');ap.add_argument('--family',choices=['uniform','spatial']);ap.add_argument('--reserve',type=float);ap.add_argument('--seed',type=int)
    a=ap.parse_args()
    if not 1<=a.workers<=2: raise ValueError('At most two simulation workers in this driver.')
    jobs=[(family,reserve,seed) for reserve in P['operating_protocol']['temperature_reserves_C'] for seed in P['search']['seeds'] for family in P['design_families'] if (a.family is None or family==a.family) and (a.reserve is None or reserve==a.reserve) and (a.seed is None or seed==a.seed)]
    results=[]
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(BASE/'search_execution.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=PROTOCOL_HASH,search_script_sha256=script_hash,workers=a.workers,chains=len(jobs),resume=a.resume))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_chain,*job,a.resume):job for job in jobs}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write_json(BASE/'search_summary.json',dict(protocol_sha256=PROTOCOL_HASH,completed_chains=len(results),planned_chains=len(jobs),chains=sorted(results,key=lambda d:d['chain_id'])))
    print('ALL_SEARCH_CHAINS_COMPLETE',len(results),flush=True)


if __name__=='__main__': main()
