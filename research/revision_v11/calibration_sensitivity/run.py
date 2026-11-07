"""Propagate the two full-precision Table S34 fits through the fixed-power stack comparison."""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'research/study_v2'))
import engine as eng
import numpy as np
from transition_diagnostics import interval_diagnostic

FITS = ROOT / 'research/revision_v9_current/numerics/selected_parameters.json'
EXPECTED = {'gamma0', 'cl_resistance_factor', 'hydration_exponent', 'tau_lb',
            'activation_energy', 'gamma_f', 'gamma_Db'}
NOMINAL_CUTOFF = 41.25
MAX_CUTOFF = 90.0


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=eng.encode, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def temperature_at(arrays, moment):
    times = arrays['dense_segment_times_s']
    choices = np.flatnonzero((times[:, 0] <= moment + 1e-10) & (moment <= times[:, 1] + 1e-10))
    if len(choices) == 0:
        return None
    row = choices[0]
    left, right = times[row]
    z = float(np.clip((moment-left)/(right-left), 0, 1))
    return np.polynomial.polynomial.polyval(z, arrays['dense_temperature_coefficients_C'][row]).tolist()


def solve(model, ti, ta, cutoff, stage, train, mesh, out, horizon=180.):
    command, _, knots, _ = eng.waveform([1.]*5, cutoff, 'constant')
    raw, final, segments, options = eng.integrate(model, horizon, ti, ta, eng.current_ramp,
                                                  command, breakpoints=knots, dense_interval=[0., horizon])
    actual_end = float(raw['time_s'])
    initial = model.initial(ti)
    arrays = eng.archive_stage(raw, final, initial, segments)
    dense = {k: arrays[k] for k in ('dense_segment_times_s', 'dense_temperature_coefficients_C', 'dense_orders')
             if k in arrays}
    reached = min(cutoff, actual_end)
    event = None
    if segments and reached > 0:
        check = interval_diagnostic(*dense.values(), [0., reached], threshold_C=1.)
        if not check['full_dense_coverage']:
            raise ValueError('Missing powered trajectory before event search')
        if check['all_means_ge_threshold_intervals_s']:
            event = check['all_means_ge_threshold_intervals_s'][0][0]
        elif temperature_at(dense, reached) and min(temperature_at(dense, reached)) >= 1.-1e-10:
            event = reached
    complete = raw['status'] == 'COMPLETE' and abs(actual_end - horizon) < 1e-8
    late = None
    if actual_end >= 180.-1e-8 and segments:
        late = interval_diagnostic(*dense.values(), [120., 180.], threshold_C=1.)
        assert late['full_dense_coverage']
    energy = raw['aux_energy_J']
    assert abs(energy-125.*reached) < 1.e-3
    if any(raw['config'][k] != getattr(model.cfg, k) for k in EXPECTED):
        raise AssertionError('Solver parameters differ from model')
    row = dict(status=raw['status'], reason=raw.get('failure_reason'), cutoff_s=cutoff, stop_s=actual_end,
               fit_C=train, mesh=mesh, stage=stage, horizon_s=horizon, complete_180s=complete and horizon==180.,
               actual_aux_J=energy, commanded_aux_J=125.*cutoff, powered_event_s=event,
               temperatures_at_powered_event_C=temperature_at(dense,event) if event is not None else None,
               temperatures_at_cutoff_C=temperature_at(dense,cutoff) if cutoff <= actual_end and segments else None,
               late_min_C=late['minimum_mea_mean_C'] if late else None,
               late_min_time_s=late['minimum_time_s'] if late else None,
               late_min_cell=late['minimum_cell_one_based'] if late else None,
               min_voltage_V=raw.get('min_voltage_V'), water_residual_relative=raw.get('water_residual_relative'),
               energy_residual_relative=raw.get('energy_residual_relative'), solver_options=options,
               parameters={k: raw['config'][k] for k in sorted(EXPECTED)})
    row['passes_1C'] = (bool(row['late_min_C'] >= 1. and row['min_voltage_V'] > .30 and
                              row['water_residual_relative'] <= 1.e-5 and row['energy_residual_relative'] <= 1.e-4)
                          if row['complete_180s'] and late else None)
    name=f'fit{abs(train)}_r{mesh}_{stage}'
    archive=out/(name+'.npz')
    np.savez_compressed(archive, **dense, times_s=arrays['time_s'], final_state_scaled=final, initial_state_scaled=initial,
                        temperature_means_C=raw['series']['temperature_C'] if 'temperature_C' in raw['series'] else
                        model.mesh.mea_means(arrays['states_scaled'][:,model.sl['T']]*model.scale[model.sl['T']]-eng.TM))
    row.update(archive=archive.name, sha256=eng.digest(archive))
    save_json(out/(name+'.json'),row)
    return row


def pair(train, mesh, out):
    fit=json.loads(FITS.read_text(encoding='utf-8'))[str(train)]['parameters']
    assert set(fit)==EXPECTED
    nominal, ti, ta, _ = eng.make_model(dict(refine=mesh,overrides={}))
    # Install every fit value after the protocol factory applies its nominal gamma_f.
    cfg=replace(nominal.cfg, **fit)
    model=eng.FuelCellModel(n_cells=5, refine=mesh, endplates=True, config=cfg,
                            backend='numba', mesh_builder=eng.build_shared)
    assert all(getattr(model.cfg,k)==v for k,v in fit.items())
    later=solve(model,ti,ta,NOMINAL_CUTOFF,'fixed',train,mesh,out)
    trace=later
    if trace['powered_event_s'] is None and trace['stop_s']>=NOMINAL_CUTOFF-1e-8:
        trace=solve(model,ti,ta,MAX_CUTOFF,'full_power_trace',train,mesh,out,horizon=MAX_CUTOFF)
    moment=trace['powered_event_s']
    early=solve(model,ti,ta,moment,'first',train,mesh,out) if moment is not None else None
    if early and early['temperatures_at_cutoff_C'] is not None:
        assert abs(min(early['temperatures_at_cutoff_C'])-1.) < 1e-4
    return dict(fit_C=train, mesh=mesh, powered_event_s=moment, powered_trace=trace['stage'],
                actual_event_means_C=trace['temperatures_at_powered_event_C'], full_power_observed_until_s=min(trace['stop_s'],trace['cutoff_s']),
                early=early, fixed=later, actual_energy_difference_J=(later['actual_aux_J']-early['actual_aux_J']) if early else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for filespec in ('fit*_r*_*.json','fit*_r*_*.npz','summary.json'):
        if any(out.glob(filespec)):
            raise FileExistsError('Choose a new output directory to reproduce the run.')
    sources=eng.source_hashes()
    for p in (FITS,Path(__file__),ROOT/'research/study_v2/transition_diagnostics.py'):
        sources[eng.relative(p)]=eng.digest(p)
    metadata=dict(sources=sources,model_conditions=dict(ambient_C=-30,initial_C=-30,heat_W_cm2_cell=1,
        cells=5,area_cm2=25,full_power_W=125,load_A_cm2='min(0.005 t,0.30)',window_s=[120,180],
        event='First time all five MEA means reach 1 C on powered trajectory; event search ends by 90 s or the actual stop.'),
        scope='Post-review propagation of two separate fits. Deterministic scenarios, no parameter probability model.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(pair,t,r,out) for t in (-20,-25) for r in (2,4)]
        for job in as_completed(jobs):
            row=job.result()
            pairs.append(row)
            print(row['fit_C'],row['mesh'],row['powered_event_s'],row['fixed']['status'],flush=True)
    metadata['pairs']=sorted(pairs,key=lambda x:(-x['fit_C'],x['mesh']))
    save_json(out/'summary.json',metadata)


if __name__ == '__main__':
    main()
