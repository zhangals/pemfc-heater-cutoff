"""Post-review structural/phase diagnostics; never changes frozen v2 records."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import engine as eng  # Sets BLAS limits before NumPy/SciPy imports.
import numpy as np
from scipy import sparse
from 模型物理 import FuelCellModel, build_shared, TM, MW, F, PREF, R
from transition_diagnostics import interval_diagnostic

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'diagnostics'
RECIPE = OUT / 'structural_phase_recipe.json'
RUNS = OUT / 'structural_phase_runs'
SCENARIOS = [
    dict(id='nominal'),
    dict(id='ep_capacity_x0p5', ep_capacity_factor=.5),
    dict(id='ep_capacity_x2', ep_capacity_factor=2.),
    dict(id='ep_bp_conductance_x0p5', ep_bp_conductance_factor=.5),
    dict(id='ep_bp_conductance_x2', ep_bp_conductance_factor=2.),
    dict(id='tau_e_x100', tau_e_factor=100.),
    dict(id='tau_e_x10000', tau_e_factor=10000.),
    dict(id='gamma_f_x10', gamma_f_factor=10.),
    dict(id='gamma_f_x100', gamma_f_factor=100.),
    dict(id='initial_liquid_saturation_0p01', initial_liquid_saturation=.01),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=eng.encode, allow_nan=False)


def source_hashes():
    paths = [Path(__file__), BASE / 'transition_diagnostics.py', BASE / 'representatives.json']
    return dict(eng.source_hashes(), **{eng.relative(p): eng.digest(p) for p in paths})


def representative():
    records = read(BASE / 'representatives.json')['representatives']
    return next(r for r in records if r['family'] == 'uniform' and r['reserve_C'] == 1)


def freeze():
    rep = representative()
    cfg = eng.make_model(dict(refine=2, overrides={}))[0].cfg
    recipe = dict(schema_version=1, frozen_utc=eng.utc_now(),
        purpose='Post-review M2/M3 structural and constitutive diagnostics, added after inspecting the original study. Not part of its frozen prespecified protocol; not physical validation or measured uncertainty intervals.',
        parent_protocol_sha256=eng.FROZEN_PROTOCOL_SHA256,
        model_source_hashes=source_hashes(), representative=rep,
        operating_conditions=dict(powers_W_cm2=[rep['design_x'][0]] * 5,
            cutoff_s=rep['design_x'][1], horizon_s=180., current='min(0.005*t,0.30) A/cm2',
            initial_C=-30., ambient_C=-30., lambda0=3., reserve_C=1., no_retuning=True),
        scenarios=SCENARIOS, refines=[2, 4], integration_count=20,
        definitions=dict(ep_capacity='Multiply EP rhocp and capacity together; geometry and conductivity unchanged. Recompute A=-diag(1/C)K.',
            ep_bp_conductance='Multiply only the two nearest-neighbour conductances whose node kinds are EP/BP. Rebuild symmetric K and A; both end-face boundary conductances are unchanged.',
            tau_e='Multiply nominal evaporation relaxation time only; not the condensation time.',
            gamma_f='Multiply the full original gamma_f factor in the coupled freeze/melt law.',
            initial_liquid='Set ml=990*eps0*0.01 kg/m3 in every resolved porous GDL/CL node, retaining all bound water and zero initial vapor/ice. Reduce reactive gas inventory by 0.99 to preserve initial reactant partial pressure in the remaining pore space.',
            ranges='Author-chosen diagnostic factors for mechanism probing, not experimentally established credible intervals.'),
        baseline_parameters=dict(tau_e_s=cfg.tau_e, gamma_f=cfg.gamma_f),
        numerics=eng.PROTOCOL['numerics'],
        output_contract=dict(dense_temperature='Native accepted-step MEA mean-temperature polynomials over [0,180], clipped to actual termination.',
            sampled_state='Native solver sample_dt=0.5 s physical states plus exact terminal state, raw signed inventories preserved.',
            minimum_windows=['[cutoff,120] s','[120,180] s'],
            local_diagnostics='CCL minimum temperature, voltage, ice and gas pressure extrema are explicitly sampled, not certified continuous extrema.',
            water='All sampled bound/vapor/liquid/ice inventories, Faraday production, outgoing water and signed closure; snapshots at nearest actual saved times and exact terminal state.',
            failures='Archive actual stop time/state and failure; requested uncovered dense-window minima are null. No retuning or replacement to obtain success.'),
        nominal_gate=dict(reference_jobs=rep['validation_jobs'], auxiliary_abs_tolerance_J=1e-5,
            electrical_abs_tolerance_J=1e-3, late_minimum_abs_tolerance_K=1e-5,
            rule='Run nominal r2/r4 first; expand only if both complete and pass all three comparisons.'),
        jobs=[dict(job_id=f'structphase_{s["id"]}_r{r}', scenario=s, refine=r)
              for s in SCENARIOS for r in [2, 4]])
    write_new(RECIPE, recipe)
    print('Frozen recipe', eng.digest(RECIPE), flush=True)


def verify_recipe():
    recipe = read(RECIPE)
    assert recipe['parent_protocol_sha256'] == eng.FROZEN_PROTOCOL_SHA256
    for p, checksum in recipe['model_source_hashes'].items():
        if eng.digest(ROOT / p) != checksum:
            raise ValueError('Source changed after diagnostic freeze: ' + p)
    return recipe


def model_for(job):
    nominal, initial_C, ambient_C, _ = eng.make_model(dict(refine=job['refine'], overrides={}))
    scenario = job['scenario']
    cfg = replace(nominal.cfg, tau_e=nominal.cfg.tau_e * scenario.get('tau_e_factor', 1.),
                  gamma_f=nominal.cfg.gamma_f * scenario.get('gamma_f_factor', 1.))

    def builder(*args):
        mesh = build_shared(*args)
        cap_factor = scenario.get('ep_capacity_factor', 1.)
        g_factor = scenario.get('ep_bp_conductance_factor', 1.)
        if cap_factor == 1. and g_factor == 1.:
            return mesh
        ep = mesh.kind == 'EP'
        rhocp, capacity, g = mesh.rhocp.copy(), mesh.capacity_J_K.copy(), mesh.conductance_W_K.copy()
        rhocp[ep] *= cap_factor
        capacity[ep] *= cap_factor
        interface = ((mesh.kind[:-1] == 'EP') & (mesh.kind[1:] == 'BP')) | ((mesh.kind[:-1] == 'BP') & (mesh.kind[1:] == 'EP'))
        assert int(interface.sum()) == 2
        g[interface] *= g_factor
        diagonal = np.r_[g, 0.] + np.r_[0., g]
        diagonal[[0, -1]] += mesh.boundary_conductance_W_K
        K = sparse.diags((-g, diagonal, -g), (-1, 0, 1), format='csr')
        result = replace(mesh, rhocp=rhocp, capacity_J_K=capacity, conductance_W_K=g,
            K_W_K=K, A_s_inv=(-sparse.diags(1. / capacity) @ K).tocsr(),
            parameter_sources=dict(mesh.parameter_sources, post_review_diagnostic=scenario))
        np.testing.assert_allclose(result.capacity_J_K, result.rhocp * result.dx * result.area_m2, rtol=1e-14)
        np.testing.assert_allclose(result.K_W_K.toarray(), result.K_W_K.T.toarray(), atol=0.)
        boundary = np.zeros(mesh.size); boundary[[0, -1]] = mesh.boundary_conductance_W_K
        np.testing.assert_allclose(result.K_W_K @ np.ones(mesh.size), boundary, atol=1e-8)
        np.testing.assert_array_equal(result.boundary_conductance_W_K, mesh.boundary_conductance_W_K)
        return result

    # Construct AFTER replacing the mesh, so the Numba closure captures the new
    # operator, capacity and constitutive coefficients rather than stale arrays.
    model = FuelCellModel(n_cells=5, refine=job['refine'], endplates=True,
                         config=cfg, backend='numba', mesh_builder=builder)
    initial = model.initial(initial_C)
    saturation = scenario.get('initial_liquid_saturation', 0.)
    if saturation:
        initial[model.sl['ml']] = 990. * model.eps0[:, 0] * saturation / model.scale[model.sl['ml']]
        initial[model.sl['ng']] *= 1. - saturation
    met = eng.solver.state_metrics(model, initial, 0.)
    assert not eng.solver.domain_failure(model, met)
    return model, initial, initial_C, ambient_C


def sampled_diagnostics(model, raw, initial, final, cutoff):
    times = np.asarray(raw['series']['time_s'])
    scaled = np.asarray(raw['states_scaled'])
    physical = scaled * model.scale
    T = physical[:, model.sl['T']]
    ccl = np.flatnonzero(model.mesh.kind == 'CCL')
    local_min = np.min(T[:, ccl], axis=1) - TM
    minV = np.min(np.asarray(raw['series']['voltage_V']), axis=1)
    ice = physical[:, model.sl['mi']] / (920. * model.eps0[:, 0])
    liquid = physical[:, model.sl['ml']] / (990. * model.eps0[:, 0])
    ice_max = np.max(ice, axis=1)
    # Match solver.state_metrics: its compact output series does not include
    # pressure, so recover this diagnostic from each native physical sample.
    pressure_nodes = model.pidx[model.an]
    pressure_field = np.empty((len(times), len(pressure_nodes)))
    for i, (t, y) in enumerate(zip(times, scaled)):
        d, z, _ = model.quantities(y, eng.current_ramp(t))
        eps = model.eps0 - d['ml'] / 990. - d['mi'] / 920.
        pgas = z['c'] * R * z['Tp']
        pv = np.maximum(d['mv'], 0.) * R * z['Tp'] / (MW * np.maximum(eps, 1e-12))
        pressure_field[i] = ((pgas + pv) / PREF - 1.)[model.an, 0]
    pressure = np.max(np.abs(pressure_field), axis=1)
    area = model.mesh.area_m2
    masses = {}
    for name in ('mb', 'mv', 'ml', 'mi'):
        dx = model.dx_b[:, 0] if name == 'mb' else model.dx_p[:, 0]
        masses[name] = area * (physical[:, model.sl[name]] @ dx)
    produced = model.nc * model.cfg.area_cm2 * physical[:, model.sl['charge']].ravel() * MW / (2 * F)
    outgoing = physical[:, model.sl['wout']].ravel()
    initial_met = eng.solver.state_metrics(model, initial, 0.)
    total = sum(masses.values())
    residual = total - initial_met['water_total_kg'] + outgoing - produced
    masks = dict(all=(times >= 0.), postcutoff_to120=(times >= cutoff) & (times <= 120.),
                 late_window=(times >= 120.) & (times <= 180.))
    windows = {}
    for name, mask in masks.items():
        idx = np.flatnonzero(mask)
        windows[name] = dict(sample_count=len(idx), complete_requested_window=bool(raw['time_s'] >= (120. if name == 'postcutoff_to120' else 180.)))
        if len(idx):
            j = idx[np.argmin(local_min[idx])]
            location = ccl[np.argmin(T[j, ccl])]
            g = idx[np.argmax(pressure[idx])]
            gnode = pressure_nodes[np.argmax(np.abs(pressure_field[g]))]
            windows[name].update(min_CCL_C=float(local_min[j]), min_CCL_time_s=float(times[j]),
                min_CCL_x_m=float(model.mesh.x[location]), min_CCL_cell_one_based=int(model.mesh.cell_ids[location] + 1),
                min_voltage_V=float(np.min(minV[idx])), max_ice_saturation=float(np.max(ice_max[idx])),
                max_liquid_saturation=float(np.max(liquid[idx])), max_anode_pressure_deviation=float(pressure[g]),
                max_anode_pressure_time_s=float(times[g]), max_anode_pressure_x_m=float(model.mesh.x[gnode]),
                max_anode_pressure_cell_one_based=int(model.mesh.cell_ids[gnode] + 1),
                max_anode_pressure_thermal_node_zero_based=int(gnode))
    regions = {'MEA':np.isin(model.mesh.kind, ['AGDL','ACL','PEM','CCL','CGDL']),
               'BP':model.mesh.kind == 'BP', 'EP':model.mesh.kind == 'EP'}
    capacities = {k:float(model.mesh.capacity_J_K[v].sum()) for k,v in regions.items()}
    capacities['total'] = float(model.mesh.capacity_J_K.sum())
    assert abs(sum(capacities[k] for k in regions) - capacities['total']) < 1e-10
    terminal = final * model.scale
    terminal_mass = {name:float(area * np.sum(terminal[model.sl[name]] * (model.dx_b[:, 0] if name == 'mb' else model.dx_p[:, 0]))) for name in masses}
    terminal_produced = float(model.nc * model.cfg.area_cm2 * terminal[model.sl['charge']][0] * MW / (2 * F))
    terminal_out = float(terminal[model.sl['wout']][0])
    terminal_residual = sum(terminal_mass.values()) - initial_met['water_total_kg'] + terminal_out - terminal_produced
    snapshots = []
    for request in [0., cutoff, 60., 120., 180.]:
        if request > raw['time_s'] + 1e-8:
            continue
        j = int(np.argmin(np.abs(times - request)))
        snapshots.append(dict(requested_time_s=request, actual_sample_time_s=float(times[j]),
            masses_kg={k:float(v[j]) for k,v in masses.items()}, produced_kg=float(produced[j]),
            outgoing_kg=float(outgoing[j]), signed_closure_kg=float(residual[j]),
            max_ice_saturation=float(ice_max[j]), max_liquid_saturation=float(np.max(liquid[j]))))
    series = dict(sampled_local_CCL_min_C=local_min, sampled_min_voltage_V=minV,
        sampled_max_ice_saturation=ice_max, sampled_max_liquid_saturation=np.max(liquid, axis=1),
        sampled_anode_pressure_deviation=pressure, water_produced_kg=produced,
        water_outgoing_kg=outgoing, water_signed_closure_kg=residual,
        **{f'water_{k}_kg':v for k,v in masses.items()})
    for region, mask in regions.items():
        series[f'sensible_{region}_J'] = (T[:, mask] - TM) @ model.mesh.capacity_J_K[mask]
    report = dict(sampling='Native solver samples; local extrema are sampled, not continuous guarantees.',
        windows=windows, heat_capacities_J_K=capacities, endplate_fraction=capacities['EP']/capacities['total'],
        snapshots=snapshots, initial_water_kg=initial_met['water_total_kg'],
        terminal_time_s=float(raw['time_s']), terminal_mass_kg=terminal_mass,
        terminal_water_produced_kg=terminal_produced, terminal_water_outgoing_kg=terminal_out,
        terminal_water_signed_closure_kg=float(terminal_residual),
        maximum_sampled_absolute_water_closure_kg=float(np.max(np.abs(residual))))
    return report, series, physical


def run_job(job):
    recipe = verify_recipe()
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (job['job_id'] + '.json')
    archive = path.with_suffix('.npz')
    lock = path.with_suffix('.lock')
    if any(p.exists() for p in (path, archive, lock)):
        raise FileExistsError('Diagnostic job already exists: ' + job['job_id'])
    started = time.perf_counter()
    with lock.open('x') as stream:
        stream.write(eng.utc_now())
    arrays = {}
    metadata = dict(job=job, job_id=job['job_id'], phase='post_review_structural_phase_diagnostic',
        recipe_sha256=eng.digest(RECIPE), parent_protocol_sha256=eng.FROZEN_PROTOCOL_SHA256,
        sources=dict(source_hashes(), **{eng.relative(RECIPE):eng.digest(RECIPE)}),
        fresh_integration=True, reused_prior_result=False, started_utc=eng.utc_now(),
        interpretation=recipe['purpose'])
    try:
        model, initial, initial_C, ambient_C = model_for(job)
        operating = recipe['operating_conditions']
        cutoff = operating['cutoff_s']
        command, _, knots, _ = eng.waveform(operating['powers_W_cm2'], cutoff, 'constant')
        raw, final, segments, solver_recipe = eng.integrate(model, 180., initial_C, ambient_C,
            eng.current_ramp, command, breakpoints=knots, initial_state=initial, dense_interval=[0., 180.])
        # The engine's generic label for an explicitly supplied initial state
        # refers to continuations; this run instead uses the documented fresh state.
        solver_recipe['initial_state_mode'] = 'Fresh nominal initial state with only recipe-prescribed liquid/gas initialization.'
        complete = raw['status'] == 'COMPLETE' and abs(raw['time_s'] - 180.) <= 1e-8
        eng.add_accounting(raw, model, initial, final, eng.current_ramp)
        sampled, series, physical = sampled_diagnostics(model, raw, initial, final, cutoff)
        raw.update(metadata, complete_180s=complete, terminal_state_available=True,
            config=model.cfg.to_dict(), initial_C=initial_C, ambient_C=ambient_C,
            powers=operating['powers_W_cm2'], cutoff_s=cutoff, solver_kwargs=solver_recipe,
            state_slices={k:[v.start,v.stop] for k,v in model.sl.items()}, sampled_diagnostics=sampled,
            audit=eng.solver.validate_constraints(raw, require_success=False))
        arrays = eng.model_arrays(model)
        arrays.update(eng.archive_stage(raw, final, initial, segments))
        arrays.update(series, sampled_states_physical=physical, initial_state_physical=initial*model.scale,
            final_state_physical=final*model.scale, mesh_rhocp_J_m3K=model.mesh.rhocp,
            mesh_conductivity_W_mK=model.mesh.k, pore_initial_porosity=model.eps0[:, 0])
        dense = {}
        for name, interval in [('postcutoff_to120',[cutoff,120.]), ('late_window',[120.,180.])]:
            if segments:
                dense[name] = interval_diagnostic(arrays['dense_segment_times_s'],
                    arrays['dense_temperature_coefficients_C'], arrays['dense_orders'], interval)
                if raw['time_s'] < interval[1] - 1e-8:
                    dense[name].update(status='UNAVAILABLE_TERMINATED_INTERVAL', full_dense_coverage=False,
                        minimum_mea_mean_C=None, minimum_time_s=None, minimum_cell_one_based=None,
                        last_upcrossing_0p01_C_s=None, all_means_ge_threshold_intervals_s=None)
            else:
                dense[name] = dict(status='UNAVAILABLE_DENSE_COVERAGE', full_dense_coverage=False, minimum_mea_mean_C=None)
        raw['dense_diagnostics'] = dense
        result = raw
    except Exception as exc:
        result = dict(metadata, status='NUMERICAL_FAILURE', failure_reason=f'{type(exc).__name__}: {exc}',
            exception_traceback=traceback.format_exc(), complete_180s=False, terminal_state_available=False)
    try:
        return eng.finish(path, archive, result, arrays, started)
    finally:
        lock.unlink(missing_ok=True)


def nominal_gate():
    recipe = verify_recipe()
    rows = []
    for refine in [2, 4]:
        new = read(RUNS / f'structphase_nominal_r{refine}.json')
        reference_id = next(x for x in recipe['nominal_gate']['reference_jobs'] if x.endswith(f'_r{refine}'))
        path = BASE / 'runs/validation' / (reference_id + '.json')
        old = read(path)
        checks = dict(auxiliary_abs_error_J=abs(new.get('aux_energy_J', float('inf')) - old['aux_energy_J']),
            electrical_abs_error_J=abs(new.get('electrical_output_J', float('inf')) - old['electrical_output_J']))
        minimum = new.get('dense_diagnostics', {}).get('late_window', {}).get('minimum_mea_mean_C')
        checks['late_minimum_abs_error_K'] = abs(minimum - old['dense_dwell']['minimum_mea_mean_C']) if minimum is not None else None
        passed = bool(new['complete_180s'] and checks['auxiliary_abs_error_J'] <= 1e-5
            and checks['electrical_abs_error_J'] <= 1e-3 and checks['late_minimum_abs_error_K'] is not None
            and checks['late_minimum_abs_error_K'] <= 1e-5)
        rows.append(dict(refine=refine, passed=passed, reference_json=eng.relative(path),
            reference_sha256=eng.digest(path), new_json_sha256=eng.digest(RUNS/f'structphase_nominal_r{refine}.json'), **checks))
    result = dict(recipe_sha256=eng.digest(RECIPE), all_passed=all(r['passed'] for r in rows), rows=rows)
    write_new(OUT / 'structural_phase_nominal_gate.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['all_passed']:
        raise RuntimeError('Nominal reproduction gate failed; remaining scenarios are blocked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--phase', choices=['nominal','remaining'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.freeze:
        freeze(); return
    if not args.phase or not 1 <= args.workers <= 4:
        parser.error('Use --freeze or --phase nominal/remaining with 1–4 workers.')
    recipe = verify_recipe()
    if args.phase == 'remaining':
        gate = read(OUT / 'structural_phase_nominal_gate.json')
        assert gate['all_passed'] and gate['recipe_sha256'] == eng.digest(RECIPE)
    jobs = [j for j in recipe['jobs'] if (j['scenario']['id'] == 'nominal') == (args.phase == 'nominal')]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_job, job):job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(result['job_id'], result['status'], result.get('time_s'), flush=True)
    if args.phase == 'nominal':
        nominal_gate()


if __name__ == '__main__':
    main()
