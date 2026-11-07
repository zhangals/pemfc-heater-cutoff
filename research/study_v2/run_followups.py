"""Run the prespecified v2 refinement, stress, waveform and baseline studies."""
from pathlib import Path
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from search import P, BASE, PROTOCOL_HASH, powers_cutoff, write_json


def existing_job(job, resume):
    path=BASE/'runs'/job['phase']/(job['job_id']+'.json')
    if not path.exists(): return None
    if not resume: raise FileExistsError(path)
    result=json.loads(path.read_text(encoding='utf-8'))
    assert result['frozen_protocol_sha256']==PROTOCOL_HASH
    stored=result['job']
    for key in ('job_id','phase','powers','cutoff_s','reserve_C','allowance_K','refine','dense','profile','overrides','policy'):
        if key in job: assert stored.get(key)==job[key], (key,stored.get(key),job[key])
    return result


def dispatch(job):
    from engine import evaluate_job, run_baseline
    return run_baseline(job) if job['phase']=='baseline' else evaluate_job(job)


def run_jobs(jobs,workers,resume):
    if not jobs: return []
    phase=jobs[0]['phase'];records=[];pending=[]
    for job in jobs:
        old=existing_job(job,resume)
        if old is None: pending.append(job)
        else: records.append(old)
    manifest=dict(phase=phase,protocol_sha256=PROTOCOL_HASH,started_utc=datetime.now(timezone.utc).isoformat(),planned_jobs=jobs,new_jobs=len(pending),recognized_completed_v2_jobs=len(records))
    write_json(BASE/f'{phase}_execution.json',manifest)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(dispatch,job):job for job in pending}
        for future in as_completed(futures):
            result=future.result();records.append(result)
            write_json(BASE/f'{phase}_progress.json',dict(completed=len(records),planned=len(jobs),last_job=futures[future]['job_id'],last_status=result.get('status'),protocol_sha256=PROTOCOL_HASH))
    manifest.update(completed_jobs=len(records),finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(BASE/f'{phase}_execution.json',manifest)
    return records


def search_chains():
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((BASE/'searches').glob('*/summary.json'))]
    expected=len(P['design_families'])*len(P['operating_protocol']['temperature_reserves_C'])*len(P['search']['seeds'])
    assert len(records)==expected, f'Need all {expected} frozen search chains, found {len(records)}'
    for d in records:
        assert d['protocol_sha256']==PROTOCOL_HASH and d['evaluations']==P['search']['evaluations_per_family_per_reserve_per_seed']
    return records


def validation_jobs():
    jobs=[]
    for chain in search_chains():
        winner=chain['winner']
        if winner is None: continue
        q,cutoff=powers_cutoff(chain['family'],winner['design_x'])
        for refine in P['numerics']['validation_refines']:
            jobs.append(dict(job_id=f'val_{chain["chain_id"]}_r{refine}',phase='validation',powers=q,cutoff_s=cutoff,reserve_C=chain['reserve_C'],allowance_K=0.,refine=refine,dense=True,profile='constant',family=chain['family'],seed=chain['seed'],source_search_job=winner['job']['job_id'],design_x=winner['design_x']))
    return jobs


def choose_representatives():
    candidates=[]
    for chain in search_chains():
        if chain['winner'] is None: continue
        rows=[]
        for refine in P['numerics']['validation_refines']:
            path=BASE/'runs'/'validation'/f'val_{chain["chain_id"]}_r{refine}.json'
            if not path.exists(): raise FileNotFoundError(path)
            d=json.loads(path.read_text(encoding='utf-8'));assert d['frozen_protocol_sha256']==PROTOCOL_HASH;rows.append(d)
        if all(d['feasible'] for d in rows):
            candidates.append(dict(family=chain['family'],reserve_C=chain['reserve_C'],seed=chain['seed'],chain_id=chain['chain_id'],design_x=chain['winner']['design_x'],planned_energy_J=chain['winner']['planned_energy_J'],source_search_job=chain['winner']['job']['job_id'],validation_jobs=[d['job']['job_id'] for d in rows]))
    selected=[];unavailable=[]
    for family in P['design_families']:
        for reserve in P['operating_protocol']['temperature_reserves_C']:
            choices=[d for d in candidates if d['family']==family and d['reserve_C']==reserve]
            if choices: selected.append(min(choices,key=lambda d:d['planned_energy_J']))
            else: unavailable.append(dict(family=family,reserve_C=reserve,reason='No selected seed candidate passed both fine grids.'))
    write_json(BASE/'representatives.json',dict(protocol_sha256=PROTOCOL_HASH,selection_rule=P['selection_and_validation']['representative_for_perturbations'],representatives=selected,unavailable=unavailable))
    return selected


def perturbation_jobs(representatives):
    cases=[]
    for key,values in P['fresh_supporting_experiments']['perturbations']['one_factor_at_a_time'].items():
        for value in values:
            if key=='h_W_m2K': over={'h':value}
            elif key=='initial_and_ambient_C': over={'initial_C':value,'ambient_C':value}
            else: over={key:value}
            cases.append((f'{key}_{value:g}'.replace('.','p').replace('-','m'),over))
    jobs=[]
    for rep in representatives:
        q,cutoff=powers_cutoff(rep['family'],rep['design_x'])
        for tag,over in cases:
            jid=f'stress_{rep["family"]}_T{rep["reserve_C"]:g}_{tag}'.replace('.','p')
            jobs.append(dict(job_id=jid,phase='perturbation',powers=q,cutoff_s=cutoff,reserve_C=rep['reserve_C'],allowance_K=0.,refine=2,dense=True,profile='constant',overrides=over,family=rep['family'],source_search_job=rep['source_search_job']))
    return jobs


def waveform_jobs(representatives):
    jobs=[]
    primary=P['operating_protocol']['primary_temperature_reserve_C']
    for rep in representatives:
        if rep['reserve_C']!=primary: continue
        q,cutoff=powers_cutoff(rep['family'],rep['design_x'])
        for refine in P['fresh_supporting_experiments']['matched_energy_waveforms']['refines']:
            for profile in P['fresh_supporting_experiments']['matched_energy_waveforms']['profiles']:
                jobs.append(dict(job_id=f'wave_{rep["family"]}_{profile}_r{refine}',phase='waveform',powers=q,cutoff_s=cutoff,reserve_C=primary,allowance_K=0.,refine=refine,dense=True,profile=profile,family=rep['family'],source_search_job=rep['source_search_job']))
    return jobs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['baseline','validation','perturbation','waveform','all'],required=True);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if not 1<=a.workers<=2: raise ValueError('At most two workers.')
    if a.phase in ('baseline','all'):
        jobs=[dict(job_id=f'{policy}_r{r}',phase='baseline',policy=policy,refine=r) for r in P['fresh_supporting_experiments']['first_crossing_and_continuation']['refines'] for policy in ('preheat','concurrent')]
        run_jobs(jobs,1,a.resume)
    if a.phase in ('validation','all'): run_jobs(validation_jobs(),a.workers,a.resume)
    if a.phase in ('perturbation','waveform','all'):
        reps=choose_representatives()
        if a.phase in ('perturbation','all'): run_jobs(perturbation_jobs(reps),a.workers,a.resume)
        if a.phase in ('waveform','all'): run_jobs(waveform_jobs(reps),a.workers,a.resume)


if __name__=='__main__': main()
