"""Compare event-triggered cutoff and sustained 1 C at identical power and load."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
V3=ROOT/'research/study_v3_diagnostics'
sys.path.insert(0,str(V3))
import run_design_recovery as base
from transition_diagnostics import interval_diagnostic

def sources():
    return {**base.structural.source_hashes(), **{base.eng.relative(p):base.eng.digest(p) for p in
              (Path(__file__),V3/'run_design_recovery.py')}}

def run(job):
    base.OUT=OUT
    base.sources=sources
    return base.run(job)

def main():
    jobs=[]
    provenance={}
    for r in (2,4):
        trace=V3/f'design_recovery/runs/capacity1p0_tau60_r{r}.npz'
        provenance[base.eng.relative(trace)]=base.eng.digest(trace)
        with np.load(trace) as z:
            d=interval_diagnostic(z['dense_segment_times_s'],z['dense_temperature_coefficients_C'],
                                  z['dense_orders'],[0.,60.],threshold_C=1.)
        assert d['full_dense_coverage']
        moment=d['all_means_ge_threshold_intervals_s'][0][0]
        assert 0<moment<60
        jobs.append(dict(id=f'first_1C_r{r}',factor=1.,cutoff=moment,refine=r))
    base.new(OUT/'protocol.json',dict(sources=sources(),inputs=provenance, jobs=jobs,
        question='Does stopping at the first 1 C crossing sustain 1 C on 120-180 s when the current ramp and heater power are identical?',
        design='On each mesh detect the first instant at which all five MEA means reach 1 C in a uniform full-power reference before its 60 s cutoff. Reintegrate from the initial state with cutoff at that instant. Compare with the existing 41.25 s passing cutoff under identical ambient temperature, area, current ramp and full power. The late-cutoff comparator is a sampled feasible policy, not a global optimum.'))
    with ProcessPoolExecutor(2) as pool:
        output=list(pool.map(run,jobs))
    pairs=[]
    for job, result in zip(jobs,output):
        r=job['refine']
        p=V3/f'design_refinement/runs/cap1p0_round4_r{r}.json'
        later=base.read(p)
        pairs.append(dict(refine=r,first_cutoff_s=job['cutoff'],first_status=result['status'],
            first_energy_J=125*job['cutoff'],
            first_late_min_C=result['dense_diagnostics']['late_window']['minimum_mea_mean_C'],
            first_transition_min_C=result['dense_diagnostics']['postcutoff_to120']['minimum_mea_mean_C'],
            sustained_cutoff_s=later['job']['cutoff'],sustained_energy_J=later['aux_energy_J'],
            sustained_min_C=later['dense_diagnostics']['late_window']['minimum_mea_mean_C'],
            energy_difference_J=later['aux_energy_J']-125*job['cutoff'],
            relative_difference_pct=100*(later['aux_energy_J']/(125*job['cutoff'])-1),
            comparator_path=base.eng.relative(p),comparator_sha256=base.eng.digest(p)))
    base.new(OUT/'summary.json',dict(pairs=pairs))
    print(json.dumps(pairs,indent=2))

if __name__=='__main__':main()

